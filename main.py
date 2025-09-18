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

index = faiss.read_index("storage/openai_index.faiss")

with open("storage/chunks_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

history = RedisHistoryManager(max_messages=40)

# Load character data
with open("character.json", "r", encoding="utf-8") as f:
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

    distances, indices = index.search(query_vector, k)

    # Build results, but ensure at least one item is returned
    results = []
    top_pairs = list(zip(distances[0], indices[0]))

    for dist, idx in top_pairs:
        if idx < 0:
            continue
        if dist < min_score:
            continue  # skip low scores
        chunk_data = metadata[idx]
        results.append({
            "score": float(dist),
            "chunk": chunk_data["content"],
            "metadata": {
                "id": chunk_data["id"],
                "title": chunk_data.get("title", ""),
                "source": chunk_data.get("source", "")
            }
        })

    if not results and top_pairs:
        dist, idx = top_pairs[0]
        if idx >= 0:
            chunk_data = metadata[idx]
            results.append({
                "score": float(dist),
                "chunk": chunk_data["content"],
                "metadata": {
                    "id": chunk_data["id"],
                    "title": chunk_data.get("title", ""),
                    "source": chunk_data.get("source", "")
                }
            })

    return results

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
    # Skip reformulation for very casual queries
    if query.lower().strip() in ["kifak", "hi", "hello", "ça va", "شلونك", "كيفك"]:
        return query
        
    model="gpt-4o"
    system_prompt = (
        "أنت مساعد متخصص في إعادة صياغة الأسئلة بطريقة مهنية ضمن نظام استرجاع المعلومات (RAG)."
        "إذا كان السؤال مكتوبًا باللهجة اللبنانية بأحرف إنجليزية، ترجم السؤال إلى العربية الفصحى بأكثر طريقة احترافية ممكنة مع الحفاظ على المهنة في التعبير."
        "ابدأ داخليًا بخطة مختصرة من ٣ إلى ٥ خطوات مفاهيمية لمعالجة كل مرحلة من مراحل السؤال، ولا تضمن هذه الخطة في النتيجة النهائية. "
        "أعد كتابة السؤال بنفس الصيغة المستخدمة من قبل المتكلم (لا تغير الضمائر أو وجهة النظر)، ولا تضف أو تحذف أي معنى جديد."
        "إذا كان السؤال واضحًا ومباشرًا، أعِد عرضه كما هو مع تحسين طفيف للأسلوب فقط."
        "الهدف هو جعل السؤال أوضح وأكثر رسمية دون تغيير معناه أو صيغة المتكلم. "
        "بعد تعديل كل سؤال، تحقق داخليًا في جملة أو جملتين أن التعديل حقق الوضوح والاحترافية دون تغيير الجوهر. "
        "اكتب فقط الصيغة النهائية للسؤال دون شرح أو خطوات."
        "الإخراج دائمًا عبارة عن السؤال النهائي المعاد صياغته فقط (جملة واحدة أو أكثر باللغة العربية الفصحى). لا تشرح أو تدرج أي تفاصيل عن العملية أو القوائم المنفذة داخليًا — الناتج النهائي هو السؤال فقط."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def generate_answer_with_history(user_id, query, retrieved_chunks, formatted_history: str) -> Tuple[str, str]:
    model = "gpt-4o"
    context = "\n\n".join([c["chunk"] for c in retrieved_chunks]) if retrieved_chunks else ""
    query_reformulated = reformulate_query(query)

    # Hierarchical system prompt with strict logic and persona embedding
    system_prompt = f'''
أنت تمثل شخصية السيد هاشم صفي الدين بدقة. يجب أن تتحدث بأسلوبه الديني الرسمي، مستخدماً المعجم والنبرة المحددة في التعليمات التالية:
{PERSONA_PREAMBLE}

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

        # Proceed with normal RAG flow for non-greetings
        prior = history.get_recent_history(user_id, max_messages=20)
        formatted = format_history_for_prompt(prior)

        query = reformulate_query(user_input)
        retrieved_chunks = search_index(query)
        answer, sources = generate_answer_with_history(user_id, user_input, retrieved_chunks=retrieved_chunks, formatted_history=formatted)
        
        # Add to history and send response
        history.add_message(user_id, "system", answer)
        
        # Combine answer and sources into one message
        if sources:
            full_response = f"{answer}\n\nالمصادر:\n{sources}"
        else:
            full_response = answer
        
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