# %pip install python-telegram-bot --quiet

import nest_asyncio
import asyncio
import random
from datetime import datetime

nest_asyncio.apply()

import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import openai
import numpy as np
import json 
import faiss
from shared_redis import r, RedisHistoryManager, format_history_for_prompt
from io import BytesIO
from functools import partial
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# Load APIs and data
load_dotenv()
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

index = faiss.read_index("data/storage/openai_index.faiss")

with open("data/storage/chunks_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

history = RedisHistoryManager(max_messages=40)

# Load character data
with open("config/character.json", "r", encoding="utf-8") as f:
    character_data = json.load(f)
    # Extract the first dictionary if it's a list
    character = character_data[0] if isinstance(character_data, list) else character_data

# Track user's last queries for response variety
user_last_queries = defaultdict(lambda: {"query": "", "count": 0})

# Functions
def get_embedding(text: str, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype='float32')

def search_index(query, k=5, min_score=0.4):
    query_vector = get_embedding(query).reshape(1, -1)
    query_vector /= np.linalg.norm(query_vector)

    distances, indices = index.search(query_vector, k * 2)  # Get more results

    # Build results with semantic search
    semantic_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        if dist < min_score:
            continue  # skip low scores
        chunk_data = metadata[idx]
        semantic_results.append({
            "score": float(dist),
            "chunk": chunk_data["content"],
            "metadata": {
                "id": chunk_data["id"],
                "title": chunk_data.get("title", ""),
                "source": chunk_data.get("source", "")
            }
        })

    # Fallback: keyword-based search if semantic search doesn't find good results
    if not semantic_results or len(semantic_results) < k:
        query_lower = query.lower()
        keyword_results = []

        for chunk_data in metadata:
            content_lower = chunk_data["content"].lower()
            title_lower = chunk_data.get("title", "").lower()

            # Check if query terms appear in content or title
            if any(term.strip() in content_lower or term.strip() in title_lower
                   for term in query_lower.split()):
                keyword_results.append({
                    "score": 0.5,  # Give keyword matches a moderate score
                    "chunk": chunk_data["content"],
                    "metadata": {
                        "id": chunk_data["id"],
                        "title": chunk_data.get("title", ""),
                        "source": chunk_data.get("source", "")
                    }
                })

        # Combine semantic and keyword results, preferring semantic
        all_results = semantic_results + keyword_results

        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(result["chunk"][:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)

        # Sort by score (higher is better)
        unique_results.sort(key=lambda x: x["score"], reverse=True)

        return unique_results[:k]

    return semantic_results[:k]

def build_persona_preamble(c) -> str:
    if isinstance(c, list):
        c = c[0] if c else {}
    elif not isinstance(c, dict):
        c = {}

    role_instructions = c.get("role_instructions")
    t = c.get("tone", {})
    tone = ", ".join(t.values())

    lex = c.get("lexicon") or {}
    inv = "\n- ".join(lex.get("invocations") or [])
    honors = "\n- ".join(lex.get("honorifics") or [])
    ashura = "\n- ".join(lex.get("ashura_register") or [])
    bins = "\n- ".join(lex.get("binaries") or [])
    values = "\n- ".join(lex.get("values") or [])

    dm_formal = "\n- ".join(lex.get("discourse_markers_formal") or [])
    dm_colloq = "\n- ".join(lex.get("discourse_markers_colloquial") or [])
    emph = "\n- ".join(lex.get("emphasis_markers") or [])
    key_terms = "\n- ".join(lex.get("key_terms") or [])

    reh = c.get("rhetorical_scaffold") or {}
    open_list = "\n- ".join(reh.get("open") or [])
    develop = "\n- ".join(reh.get("develop") or [])
    evidence = "\n- ".join(reh.get("evidence") or [])
    application = "\n- ".join(reh.get("application") or [])
    closure = "\n- ".join(reh.get("closure") or [])

    pacing = c.get("response_pacing", {})
    response_pacing = ", ".join(pacing.values())

    greetings = "\n- ".join(c.get("greeting_templates") or [])
    closing = "\n- ".join(c.get("closing_templates") or [])
    condolences = "\n- ".join(c.get("condolence_templates") or [])

    q = c.get("quote_frames", {})
    quote_frames = ", ".join(q.values())

    do = "\n- ".join(c.get("do") or [])
    dont = "\n- ".join(c.get("dont") or [])

    snippet = c.get("style_snippets", {})
    style_snippets = ", ".join(snippet.values())

    micro = c.get("micro_templates", {})
    micro_templates = ", ".join(micro.values())

    topics = "\n- ".join(c.get("topics") or [])

    tk = c.get("topics_knowledge") or {}
    personal_section = ""
    other_topics_sections = []
    if isinstance(tk, dict):
        for name, data in tk.items():
            is_personal = isinstance(name, str) and "سيرة السيد هاشم صفيّ الدين" in name
            highlights = []
            points = []
            use_with = None
            if isinstance(data, dict):
                highlights = data.get("highlights") or []
                points = data.get("points") or []
                use_with = data.get("use_with")
            if is_personal:
                lines = []
                if points:
                    lines.extend(points)
                elif highlights:
                    lines.extend(highlights)
                else:
                    lines.extend([f"{k}: {v}" for k, v in data.items()])
                personal_section = "\n".join(["السيرة الشخصية:"] + [f"- {x}" for x in lines])
            else:
                lines = []
                if highlights:
                    lines.extend(highlights)
                elif points:
                    lines.extend(points)
                else:
                    lines.extend([f"{k}: {v}" for k, v in data.items()])
                section = "\n".join([name + ":"] + [f"- {x}" for x in lines] + ([f"- use_with: {use_with}"] if use_with else []))
                other_topics_sections.append(section)
    topics_knowledge_personal = personal_section
    topics_knowledge_other = "\n\n".join(other_topics_sections)

    cu = c.get("contextual_usage") or {}
    contextual_usage = "\n".join(["قيود الاستخدام السياقي (إلزامي):"] + [f"- {k}: {v}" for k, v in cu.items()])

    return (
        f"الشخصية: {c.get('name','')}\n"
        f"الغرض: {c.get('purpose','')}\n"
        f"تعليمات الدور: {role_instructions}\n"
        f"النبرة: {tone}\n"
        f"الافتتاحيات:\n- {inv}\n"
        f"الألقاب:\n- {honors}\n"
        f"سجل عاشورائي:\n- {ashura}\n"
        f"الثنائيات:\n- {bins}\n"
        f"القيم:\n- {values}\n"
        f"روابط الخطاب (فصحى):\n- {dm_formal}\n"
        f"روابط الخطاب (عامية):\n- {dm_colloq}\n"
        f"علامات التأكيد:\n- {emph}\n"
        f"مصطلحات مفتاحية:\n- {key_terms}\n"
        f"التمهيد البلاغي:\n- {open_list}\n"
        f"التطوير البلاغي:\n- {develop}\n"
        f"أمثلة وأدلة:\n- {evidence}\n"
        f"تطبيق البلاغة:\n- {application}\n"
        f"الإغلاق البلاغي:\n- {closure}\n"
        f"إيقاع الاستجابة: {response_pacing}\n"
        f"قوالب الترحيب:\n- {greetings}\n"
        f"قوالب الختام:\n- {closing}\n"
        f"قوالب التعزية:\n- {condolences}\n"
        f"أطر الاقتباس: {quote_frames}\n"
        f"افعل:\n- {do}\n"
        f"لا تفعل:\n- {dont}\n"
        f"مقتطفات أسلوبية: {style_snippets}\n"
        f"قوالب دقيقة: {micro_templates}\n"
        f"الموضوعات:\n- {topics}\n"
        f"{topics_knowledge_personal}\n\n{topics_knowledge_other}\n"
        f"{contextual_usage}\n"
        f"تنبيه: الالتزام بما سبق إلزامي في كل إجابة."
    )

PERSONA_PREAMBLE = build_persona_preamble(character)

def is_greeting(text):
    greeting_phrases = [
        "السلام عليكم", "سلام", "مرحبا", "اهلا", "hello", "hi", 
        "صباح الخير", "مساء الخير", "كيفك", "kifak", "bonjour", 
        "اهلاً", "أهلا", "مرحباً", "كيف الحال", "شلونك", "شونك",
        "ça va", "how are you", "what's up", "كيفك", "شخبارك"
    ]
    text_lower = text.lower().strip()
    
    # Check for exact matches or contained phrases
    for phrase in greeting_phrases:
        if phrase == text_lower or phrase in text_lower:
            return True
    
    # Check for very short messages that are likely greetings
    if len(text_lower.split()) <= 2 and any(word in text_lower for word in ['كيف', 'شلون', 'شو', 'اهل', 'مرحب']):
        return True
        
    return False

def get_appropriate_greeting(user_input, is_casual=False):
    """Return context-appropriate greeting based on user input and time of day"""
    
    # Extract greeting templates from character
    greeting_templates = character.get("greeting_templates", [])
    
    # Time-based greetings
    current_hour = datetime.now().hour
    time_based = []
    
    if 5 <= current_hour < 12:
        time_based = ["صباح الخير", "صباح النور"]
    elif 12 <= current_hour < 18:
        time_based = ["مساء الخير", "مساء النور"]
    else:
        time_based = ["مساء الخير", "تصبح على خير"]
    
    # Casual greetings for informal inputs
    casual_greetings = [
        "أهلاً وسهلاً", "مرحباً", "أهلاً بك", "حياك الله",
        "الله يسلمك", "على الرحب والسعة"
    ]
    
    # Choose based on context
    if is_casual:
        options = casual_greetings + time_based
    else:
        options = greeting_templates + time_based
    
    return random.choice(options)

def detect_topic_type(query):
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['دين', 'اسلام', 'قرآن', 'حديث', 'صلاة', 'صوم', 'عبادة']):
        return "religious"
    elif any(word in query_lower for word in ['تاريخ', 'قصة', 'سيرة', 'حرب', 'مقاومة', 'حدث']):
        return "historical"
    elif any(word in query_lower for word in ['سؤال', 'استفسار', 'رأي', 'نصيحة', 'فتوى']):
        return "advice"
    else:
        return "casual"

def reformulate_query(query):
    """
    Normalize query to Modern Standard Arabic while preserving original intent.
    For retrieval purposes, we want to keep the core terms but in proper Arabic.
    """
    # Skip reformulation for very casual queries
    if query.lower().strip() in ["kifak", "hi", "hello", "ça va", "شلونك", "كيفك"]:
        return query

    # Simple normalization rules for common Lebanese dialect to Fusha
    query = query.replace("شو", "ما")
    query = query.replace("شلون", "كيف")
    query = query.replace("كيفك", "كيف حالك")
    query = query.replace("هاي", "هذا")
    query = query.replace("هيك", "هكذا")
    query = query.replace("بدي", "أريد")
    query = query.replace("بدك", "تريد")
    query = query.replace("بدو", "يريد")

    # If it's a single word or very short, don't change it much
    words = query.split()
    if len(words) <= 3:
        return query

    # For longer queries, use LLM to normalize while preserving intent
    model="gpt-4o"
    system_prompt = (
        "مهمتك: تحويل النص إلى العربية الفصحى مع الحفاظ على المعنى الأصلي تماماً."
        "لا تغيّر هيكل السؤال أو تضيف معلومات جديدة."
        "ركز على تصحيح القواعد والعامية فقط."
        "الإخراج: النص المصحح فقط."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except:
        # Fallback to original query if LLM fails
        return query

def generate_answer_with_history(user_id, query, retrieved_chunks, formatted_history: str, previous_feedback: str = "", iteration: int = 1) -> Tuple[str, str]:
    model = "gpt-4o"
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks]) if retrieved_chunks else ""
    query_reformulated = reformulate_query(query)

    # Enhanced system prompt with escalating strictness based on iteration
    strictness_level = min(iteration, 3)  # Cap at level 3 for maximum strictness
    
    # Base strictness messages
    strictness_messages = {
        1: "يجب الالتزام التام بأسلوب السيد هاشم صفي الدين. أي انحراف عن الشخصية سيؤدي إلى إعادة المحاولة.",
        2: "هذا التحذير الثاني: يجب الالتزام الصارم بأسلوب السيد هاشم صفي الدين. أي إجابة غير مناسبة للشخصية ستُرفض تماماً وتُعاد المحاولة.",
        3: "تحذير نهائي: يجب الالتزام المطلق بأسلوب السيد هاشم صفي الدين. أي إجابة لا تتطابق تماماً مع الشخصية ستُرفض نهائياً وتُعاد المحاولة مع عقوبات أشد."
    }
    
    strictness_message = strictness_messages.get(strictness_level, strictness_messages[3])
    
    system_prompt = f'''
أنت تمثل شخصية السيد هاشم صفي الدين بدقة. يجب أن تتحدث بأسلوبه الديني الرسمي، مستخدماً المعجم والنبرة المحددة في التعليمات التالية:
{PERSONA_PREAMBLE}

{strictness_message}

قواعد التصرف:
1. للأسئلة العامة أو التحية (مثل السلام عليكم، كيف حالك، من أنت): أجب مباشرة بالأسلوب المحدد دون استرجاع مقاطع.

2. للأسئلة الأخرى:
   - إذا لم يتم استرجاع مقاطع، اعترف بعدم وجود معلومات كافية.
   - استند فقط إلى المقاطع المسترجعة، لا تخترع معلومات.
   - إذا لم يكن الجواب في المقاطع، اعترف بحدود المعرفة.

منهجية الإجابة:
1. استخرج المحتوى من المقاطع فقط.
2. صغ الإجابة بأسلوب السيد هاشم: استخدم الافتتاحيات، الألقاب، الثنائيات، القيم، روابط الخطاب الفصحى والعامية حسب السياق، علامات التأكيد، المصطلحات المفتاحية، التمهيد البلاغي، أمثلة وأدلة، تطبيق، إغلاق.
3. لا تذكر عملية الاسترجاع أو تفاصيل تقنية.
4. أجب بالعربية الفصحى فقط.
5. كن موجزاً ودقيقاً، مع إيقاع متوسط يميل للاختصار مع جمل إيقاعية.
6. التزم بالقيود السياقية: استخدم الافتتاحيات في الخطب أو الردود المطولة، التعزية فقط لعاشوراء، الختام في المقاطع الخطابية.

{"التحسينات السابقة: " + previous_feedback if previous_feedback else ""}
'''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"\n\nالرسائل السابقة:\n{formatted_history}"},
        {"role": "user", "content": f"السياق:\n\n{context}\n\nالسؤال: {query_reformulated}"},
    ]
    
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    answer_text = response.choices[0].message.content.strip()

    # Build unique citations list from retrieved chunks' metadata
    citations = []
    if retrieved_chunks:
        seen = set()
        for item in retrieved_chunks:
            md = item.get("metadata", {})
            title = (md.get("title") or "").strip()
            source = (md.get("source") or "").strip()
            key = (title, source)
            if (title or source) and key not in seen:
                seen.add(key)
                if title and source:
                    citations.append(f"- {title} — {source}")
                elif title:
                    citations.append(f"- {title}")
                else:
                    citations.append(f"- {source}")

    sources_text = "\n".join(citations) if citations else ""
    
    return answer_text, sources_text

def validate_response_and_sources(query, answer_text, sources_text, formatted_history, retrieved_chunks):
    """
    LLM-driven validation of response quality and source appropriateness.
    Returns: (should_include_sources, is_answer_appropriate, is_persona_consistent,
              is_islamic_valid, is_format_correct, validation_feedback, should_rerun)
    """
    model = "gpt-4o"

    validation_prompt = f"""
أنت محكم دقيق لجودة إجابات chatbot إسلامي يمثل شخصية السيد هاشم صفي الدين.

السؤال الأصلي: {query}
الإجابة المقدمة: {answer_text}
المصادر المقترحة: {sources_text}
المحادثة السابقة: {formatted_history}

معايير التقييم الصارمة (يجب الالتزام بجميعها):

1. **الشخصية والأسلوب**:
   - هل الإجابة بأسلوب السيد هاشم صفي الدين تماماً؟
   - هل تحتوي على الافتتاحيات الدينية المناسبة؟
   - هل تستخدم المصطلحات والعبارات التي يستخدمها السيد هاشم؟
   - هل تتجنب أي أسلوب حديث أو غير ديني؟

2. **المحتوى الإسلامي**:
   - إذا كان السؤال دينياً، هل تحتوي على آية قرآنية أو حديث نبوي؟
   - هل المحتوى دقيق إسلامياً وغير منحرف؟
   - هل تتضمن الدروس والعبر المناسبة؟
   - هل تتجنب الإجابات العامة غير المرتبطة بالسيد هاشم؟

3. **الشكل والتنسيق**:
   - هل تبدأ بالتحية المناسبة؟
   - هل تنتهي بدعاء أو خاتمة دينية؟
   - هل تتبع هيكل: تحية → شرح → درس → خاتمة؟
   - هل الإجابة بالعربية الفصحى فقط؟

4. **المصادر**:
   - إذا كانت الإجابة مبنية على كلام السيد هاشم الفعلي، يجب تضمين المصادر
   - إذا كانت إجابة عامة أو تحية، لا تحتاج مصادر
   - المصادر يجب أن تكون من مصادر السيد هاشم الفعلية

معايير الرفض الفوري:
- إجابة عامة غير مرتبطة بالسيد هاشم
- عدم استخدام الأسلوب الديني المناسب
- عدم وجود آية أو حديث في السؤال الديني
- أسلوب حديث أو غير رسمي
- عدم اتباع التنسيق الصحيح

أجب بصيغة JSON فقط:
{{
    "include_sources": true/false,
    "answer_appropriate": true/false,
    "persona_consistent": true/false,
    "islamic_content_valid": true/false,
    "format_correct": true/false,
    "feedback": "تعليقاتك التفصيلية على الإجابة",
    "should_rerun": true/false
}}
"""

    try:
        messages = [
            {"role": "system", "content": validation_prompt}
        ]

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )

        result = response.choices[0].message.content.strip()

        # Parse JSON response
        import json
        validation_result = json.loads(result)

        return (
            validation_result.get("include_sources", False),
            validation_result.get("answer_appropriate", True),
            validation_result.get("persona_consistent", True),
            validation_result.get("islamic_content_valid", True),
            validation_result.get("format_correct", True),
            validation_result.get("feedback", ""),
            validation_result.get("should_rerun", False)
        )

    except Exception as e:
        # Fallback in case of parsing error
        print(f"Validation error: {e}")
        return False, True, "خطأ في التحليل", False

def generate_validated_response(user_id, query, formatted_history: str, max_iterations: int = 3) -> str:
    """
    Generate a validated response with iterative improvement and LLM-driven quality control.
    """
    previous_feedback = ""
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Retrieve relevant chunks
        query_reformulated = reformulate_query(query)
        retrieved_chunks = search_index(query_reformulated)

        # Generate initial response
        answer_text, sources_text = generate_answer_with_history(
            user_id, query, retrieved_chunks, formatted_history, previous_feedback, iteration
        )

        # Validate the response
        (should_include_sources, is_appropriate, is_persona_consistent,
         is_islamic_valid, is_format_correct, feedback, should_rerun) = validate_response_and_sources(
            query, answer_text, sources_text, formatted_history, retrieved_chunks
        )

        # Check all validation criteria
        validation_passed = (is_appropriate and is_persona_consistent and
                           is_islamic_valid and is_format_correct)

        # If validation passes and no rerun needed, return the response
        if not should_rerun and is_appropriate:
            final_sources = sources_text if should_include_sources else ""
            if final_sources:
                return f"{answer_text}\n\nالمصادر:\n{final_sources}"
            else:
                return answer_text

        # If this is the last iteration or validation failed critically, return anyway
        if iteration >= max_iterations:
            final_sources = sources_text if should_include_sources else ""
            if final_sources:
                return f"{answer_text}\n\nالمصادر:\n{final_sources}"
            else:
                return answer_text

        # Prepare feedback for next iteration
        previous_feedback = f"""
        الإجابة السابقة: {answer_text}
        تعليقات التحسين: {feedback}
        هل كانت الإجابة مناسبة: {'نعم' if is_appropriate else 'لا'}
        هل يجب تضمين المصادر: {'نعم' if should_include_sources else 'لا'}

        يرجى تحسين الإجابة بناءً على هذه التعليقات في المحاولة التالية.
        """

    # Fallback
    return "عذراً، حدث خطأ في معالجة الاستعلام. يرجى المحاولة مرة أخرى."

# Handlers
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history.clear(user_id)
    await update.message.reply_text("تم مسح تاريخ المحادثة.")


async def handle_message(type, update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        if type == "voice":
            voice_file = await update.message.voice.get_file()
            audio_bytes = BytesIO()
            await voice_file.download_to_memory(out=audio_bytes)
            audio_bytes.seek(0)

            transcription = elevenlabs.speech_to_text.convert(
                file=audio_bytes,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="ara",
                diarize=True,
            )
            user_input = transcription.text

        elif type == "text":
            user_input = update.message.text
        else:
            await update.message.reply_text("الرجاء إرسال رسالة صوتية أو نصية فقط، الصيغة المُرسلة غير مدعومة.")
            return

        # Check for repeated queries
        if user_last_queries[user_id]["query"] == user_input:
            user_last_queries[user_id]["count"] += 1
        else:
            user_last_queries[user_id] = {"query": user_input, "count": 1}
        
        # Add user message to history
        history.add_message(user_id, "user", user_input)

        # Check if the message is a greeting
        if is_greeting(user_input):
            is_casual = any(word in user_input.lower() for word in ['kifak', 'hi', 'hello', 'ça va', 'شلونك', 'كيفك'])
            chosen_greeting = get_appropriate_greeting(user_input, is_casual)
            
            # Add natural follow-up
            follow_ups = [
                "كيف يمكنني مساعدتك اليوم؟",
                "ما الذي تريد معرفته؟",
                "هل لديك سؤال أو استفسار؟",
            ]
            
            response = f"{chosen_greeting} {random.choice(follow_ups)}"
            history.add_message(user_id, "system", response)
            await update.message.reply_text(response)
            return

        # Proceed with validated response generation
        prior = history.get_recent_history(user_id, max_messages=20)
        formatted = format_history_for_prompt(prior)

        # Generate validated response with iterative improvement
        full_response = generate_validated_response(user_id, user_input, formatted)

        # Add to history and send response
        history.add_message(user_id, "system", full_response)
        await update.message.reply_text(full_response)

    except Exception as e:
        await update.message.reply_text(f"حدث خطآ، حاول مجددا. {e}")


# Build Bot
app = Application.builder().token(telegram_token).build()
# Commands
app.add_handler(CommandHandler("clear", clear_history))
# Messages
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_message, "text")))
app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, partial(handle_message, "voice")))

if __name__ == "__main__":
    print("Bot is running")
    app.run_polling()
    print("Bot has stopped.")