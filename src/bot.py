# bot.py
"""
Production-ready async Telegram chatbot pipeline for Sayed Hashem Safieddine persona.
Uses LLM at every stage with strict validation and persona fidelity.

Environment Variables Required:
- OPENAI_API_KEY: OpenAI API key for LLM calls
- TELEGRAM_BOT_TOKEN: Telegram bot token
- ELEVENLABS_API_KEY: For speech-to-text (optional)

Run with: python bot.py
"""

import asyncio
import json
import logging
import os
import random
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import faiss
import nest_asyncio
import numpy as np
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from functools import partial

from shared_redis import RedisHistoryManager, format_history_for_prompt
from prompts import (
    MASTER_PROMPT,
)
from arabic_utils import diacritize_arabic_text, calculate_arabic_diacritics_coverage, ensure_arabic_diacritization
from utils.diacritizer import apply_diacritics, compute_diacritics_coverage

# Determine project root (parent directory of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load FAISS index and metadata
with open(os.path.join(PROJECT_ROOT, "data", "storage", "chunks_metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load FAISS indexes (support multiple for performance scaling)
indexes = {}
categories = ["tafsir", "hadith", "speeches", "history", "politics", "general"]
for category in categories:
    index_path = os.path.join(PROJECT_ROOT, "data", "storage", f"{category}_index.faiss")
    if os.path.exists(index_path):
        indexes[category] = faiss.read_index(index_path)
    else:
        indexes[category] = None

# Main index as fallback
main_index_path = os.path.join(PROJECT_ROOT, "data", "storage", "openai_index.faiss")
if os.path.exists(main_index_path):
    main_index = faiss.read_index(main_index_path)
    for category in categories:
        if indexes[category] is None:
            indexes[category] = main_index
else:
    main_index = None

# For backward compatibility
index = main_index

# Load character data
with open(os.path.join(PROJECT_ROOT, "config", "character.json"), "r", encoding="utf-8") as f:
    character_data = json.load(f)
character = character_data[0] if isinstance(character_data, list) else character_data

# Load personality instructions
with open(os.path.join(PROJECT_ROOT, "config", "personality_instructions.json"), "r", encoding="utf-8") as f:
    personality_data = json.load(f)

# Build persona preamble
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

    # Build personality instructions section
    pi = personality_data.get("personality_instructions", {})
    personality_section = "\n\nتعليمات الشخصية التفصيلية:\n"
    
    for category, features in pi.items():
        personality_section += f"\n{category.replace('_', ' ').title()}:\n"
        if isinstance(features, dict):
            for subcat, items in features.items():
                if isinstance(items, list):
                    personality_section += f"- {subcat.replace('_', ' ').title()}: {', '.join(items)}\n"
                else:
                    personality_section += f"- {subcat.replace('_', ' ').title()}: {items}\n"
        elif isinstance(features, str):
            personality_section += f"- {features}\n"

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
        f"{personality_section}\n"
        f"تنبيه: الالتزام بما سبق إلزامي في كل إجابة."
    )

PERSONA_PREAMBLE = build_persona_preamble(character)

# History manager
history = RedisHistoryManager(max_messages=40)

# Response cache
response_cache = {}

def get_cache_key(query: str, chat_history: str) -> str:
    import hashlib
    key = f"{query}:{chat_history}"
    return hashlib.md5(key.encode()).hexdigest()

def get_cached_response(query: str, chat_history: str) -> str:
    key = get_cache_key(query, chat_history)
    return response_cache.get(key)

def set_cached_response(query: str, chat_history: str, response: str):
    key = get_cache_key(query, chat_history)
    response_cache[key] = response

# Constants
MAX_ITERATIONS = 3
MAX_RESPONSE_LENGTH = 2000

async def get_embedding(text: str) -> np.ndarray:
    try:
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype='float32')
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return np.zeros(1536, dtype='float32')  # Return zero vector if error



async def classify_llm(query: str) -> List[str]:
    # Simple keyword-based classification
    query_lower = query.lower()
    corpora = []
    if "سورة" in query_lower or "آية" in query_lower:
        corpora.append("tafsir")
    if "حديث" in query_lower:
        corpora.append("hadith")
    if "خطبة" in query_lower or "محاضرة" in query_lower:
        corpora.append("speeches")
    if "تاريخ" in query_lower:
        corpora.append("history")
    if "سياسة" in query_lower:
        corpora.append("politics")
    return corpora or ["general"]

async def retrieve_chunks(query: str, corpora: List[str], k: int = 10) -> List[Dict]:
    try:
        query_vector = await get_embedding(query)
        query_vector = query_vector.reshape(1, -1)
        query_vector /= np.linalg.norm(query_vector)

        # Choose index based on corpora (use specific if single category, else main)
        if corpora and len(corpora) == 1 and corpora[0] in indexes:
            selected_index = indexes[corpora[0]]
        else:
            selected_index = main_index

        distances, indices = selected_index.search(query_vector, k * 2)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or dist > 0.4:
                continue
            chunk_data = metadata[idx]
            score = float(dist)
            # Weight for corpus match - boost matching chunks
            if corpora and any(corpus in chunk_data.get("source", "").lower() for corpus in corpora):
                score *= 0.7  # Reduce distance for better ranking
            results.append({
                "score": score,
                "chunk": chunk_data["content"],
                "metadata": chunk_data
            })
        
        # Sort by score (lower is better) and take top k
        results.sort(key=lambda x: x["score"])
        return results[:k]
    except Exception as e:
        logger.error(f"Error in retrieve_chunks: {e}")
        return []

def identify_used_sources(response: str, retrieved_chunks: List[Dict]) -> List[Dict]:
    """Identify which sources were actually referenced in the response."""
    used_sources = []

    for chunk in retrieved_chunks:
        metadata = chunk.get("metadata", {})
        content = chunk.get("chunk", "")

        # Check if the response contains phrases that indicate this source was used
        # Look for direct quotes, specific references, or key phrases from the chunk
        content_phrases = content.split('.')[:3]  # First few sentences
        source_indicators = [
            metadata.get("title", ""),
            metadata.get("source", ""),
            metadata.get("author", "")
        ]

        # Check if response contains significant phrases from this chunk
        significant_matches = 0
        for phrase in content_phrases:
            phrase = phrase.strip()
            if len(phrase) > 20:  # Only check substantial phrases
                # Look for partial matches (3+ consecutive words)
                words = phrase.split()
                for i in range(len(words) - 2):
                    trigram = ' '.join(words[i:i+3])
                    if trigram in response:
                        significant_matches += 1
                        break

        # If we found significant matches or direct source references
        if significant_matches > 0 or any(indicator in response for indicator in source_indicators if indicator):
            used_sources.append({
                "title": metadata.get("title", "مصدر غير محدد"),
                "source": metadata.get("source", ""),
                "author": metadata.get("author", ""),
                "url": metadata.get("url", ""),
                "date": metadata.get("date", "")
            })

    return used_sources

def generate_followup_question(query: str, response: str, chat_history: str, context_analysis: Dict = None) -> str:
    """Generate a contextual follow-up question to encourage deeper conversation."""
    if context_analysis is None:
        context_analysis = analyze_conversation_context(chat_history, query)

    # Different follow-up strategies based on context
    if context_analysis["emotional_state"] == "negative":
        followups = [
            "ما الذي يقلقك أكثر في هذا الموضوع؟",
            "هل تريد أن نتحدث عن طرق للتعامل مع هذا الشعور؟",
            "كيف يمكنني مساعدتك في تخفيف هذا الهم؟"
        ]
    elif context_analysis["emotional_state"] == "confused":
        followups = [
            "هل أوضحت الإجابة ما كنت تريد معرفته؟",
            "أي جزء من الإجابة يحتاج إلى توضيح أكثر؟",
            "هل تريد أن أشرح بطريقة مختلفة؟"
        ]
    elif context_analysis["repeated_questions"]:
        followups = [
            "هل تجد صعوبة في فهم الإجابات السابقة؟",
            "ما الجانب الذي ما زال غامضاً بالنسبة لك؟",
            "هل تريد أن نركز على جانب معين من الموضوع؟"
        ]
    elif context_analysis["depth"] > 10:  # Deep conversation
        followups = [
            "كيف ترى تطبيق هذا في حياتك اليومية؟",
            "ما الخطوة التالية التي ستتخذها؟",
            "هل هناك جوانب أخرى تريد استكشافها؟"
        ]
    else:  # Normal conversation
        followups = [
            "كيف تشعر تجاه ما قلته؟",
            "هل لديك أسئلة إضافية حول هذا الموضوع؟",
            "ما رأيك في هذا الجانب من الدين الإسلامي؟"
        ]

    # Topic-specific follow-ups
    if "prayer" in context_analysis["topics_discussed"]:
        followups.extend([
            "كيف هو حال صلاتك هذه الأيام؟",
            "هل تواجه صعوبة في التركيز أثناء الصلاة؟"
        ])
    elif "quran" in context_analysis["topics_discussed"]:
        followups.extend([
            "ما السورة التي تحب تلاوتها أكثر؟",
            "هل تطبق تعاليم القرآن في حياتك اليومية؟"
        ])

    return random.choice(followups)

def generate_fallback_response(query: str, error_type: str) -> str:
    """Generate a helpful fallback response when main generation fails."""
    base_response = "أعوذ بالله من شر الشيطان الرجيم. بسم الله الرحمن الرحيم.\n\n"

    if "retrieval" in error_type:
        base_response += "عذراً، واجهت صعوبة في البحث عن المعلومات المطلوبة. "
    elif "generation" in error_type:
        base_response += "عذراً، واجهت صعوبة في صياغة الإجابة. "
    else:
        base_response += "عذراً، حدث خطأ تقني. "

    # Provide helpful alternatives based on query type
    query_lower = query.lower()
    if any(word in query_lower for word in ["صلاة", "عبادة", "قرآن", "حديث"]):
        base_response += "لكن يمكنني مساعدتك في مواضيع أخرى متعلقة بالعبادات والقرآن الكريم. ما السؤال الذي يدور في بالك؟"
    elif any(word in query_lower for word in ["عاشوراء", "حسين", "كربلاء"]):
        base_response += "لكن يمكنني الحديث عن قيم عاشوراء ودروسها العظيمة. هل تريد معرفة المزيد عن هذه المناسبة المباركة؟"
    elif any(word in query_lower for word in ["أخلاق", "أدب", "سلوك"]):
        base_response += "لكن يمكنني مساعدتك في فهم الأخلاق الإسلامية والسلوك القويم. ما الجانب الذي يهمك؟"
    else:
        base_response += "لكن يمكنني مساعدتك في أسئلة أخرى متعلقة بالدين والحياة الإسلامية. ما الذي يشغل بالك؟"

    return base_response

def analyze_conversation_context(chat_history: str, query: str) -> Dict[str, any]:
    """Analyze conversation context to provide more intelligent responses."""
    context_info = {
        "depth": 0,
        "emotional_state": "neutral",
        "topics_discussed": [],
        "repeated_questions": False,
        "follow_up_needed": False
    }

    if not chat_history:
        return context_info

    lines = chat_history.split('\n')
    context_info["depth"] = len([line for line in lines if line.strip()])

    # Analyze emotional indicators
    emotional_words = {
        "positive": ["الحمد", "شكراً", "ممتاز", "جميل", "رائع"],
        "negative": ["مشكلة", "صعب", "قلق", "حزين", "غاضب"],
        "confused": ["لا أفهم", "مش عارف", "كيف", "لماذا", "ليش"],
        "seeking": ["أريد", "أحتاج", "أبحث عن", "أسأل عن"]
    }

    for emotion, words in emotional_words.items():
        if any(word in chat_history for word in words):
            context_info["emotional_state"] = emotion
            break

    # Extract topics (simple keyword analysis)
    topics = []
    if any(word in chat_history for word in ["صلاة", "عبادة"]):
        topics.append("prayer")
    if any(word in chat_history for word in ["قرآن", "سورة", "آية"]):
        topics.append("quran")
    if any(word in chat_history for word in ["عاشوراء", "حسين", "كربلاء"]):
        topics.append("ashura")
    if any(word in chat_history for word in ["أخلاق", "أدب", "سلوك"]):
        topics.append("ethics")

    context_info["topics_discussed"] = topics

    # Check for repeated questions
    user_queries = [line for line in lines if "user:" in line.lower()]
    if len(user_queries) > 2:
        recent_queries = user_queries[-3:]
        # Simple check for similar queries
        if len(set(recent_queries)) < len(recent_queries):
            context_info["repeated_questions"] = True

    # Determine if follow-up is needed
    context_info["follow_up_needed"] = (
        context_info["emotional_state"] in ["negative", "confused"] or
        context_info["repeated_questions"] or
        len(topics) > 2  # Deep conversation
    )

    return context_info

async def generate_validated_response(user_id: int, query: str, chat_history: str) -> str:
    # Check cache
    cached = get_cached_response(query, chat_history)
    if cached:
        return cached
    
    # Check for name-related queries
    query_lower = query.lower()
    if "shu esme" in query_lower or "what is my name" in query_lower or "اسمي" in query_lower or "ana shu esme" in query_lower:
        user_profile = history.get_user_profile(user_id)
        name = user_profile.get("name")
        if name:
            response = f"أعوذ بالله من شر الشيطان الرجيم. بسم الله الرحمن الرحيم.\n\nاسمك {name}، يا {name}. كيف يمكنني خدمتك اليوم؟"[:MAX_RESPONSE_LENGTH]
            set_cached_response(query, chat_history, response)
            return response
    
    # Handle special cases (simplified)
    if "مرحبا" in query_lower or "hello" in query_lower or "hi" in query_lower:
        user_profile = history.get_user_profile(user_id)
        name = user_profile.get("name", "")
        greeting = "السلام عليكم ورحمة الله وبركاته."
        if name:
            greeting += f" أهلاً وسهلاً بك، يا {name}."
        else:
            greeting += " أهلاً وسهلاً بك."
        greeting += " كيف يمكنني خدمتك اليوم؟"
        response = greeting[:MAX_RESPONSE_LENGTH]
        set_cached_response(query, chat_history, response)
        return response
    elif "من أنت" in query_lower or "who are you" in query_lower or "tell me about yourself" in query_lower:
        user_profile = history.get_user_profile(user_id)
        name = user_profile.get("name", "")
        identity_response = "أعوذ بالله من شر الشيطان الرجيم. بسم الله الرحمن الرحيم.\n\nأنا هاشم صفيّ الدين، خادم لأهل البيت سلام الله عليهم. أقدم لكم النصائح والمعلومات المستندة إلى القرآن الكريم والحديث الشريف وتراث أهل البيت."
        if name:
            identity_response += f" يا {name}."
        identity_response += " كيف يمكنني خدمتكم اليوم؟"
        response = identity_response[:MAX_RESPONSE_LENGTH]
        set_cached_response(query, chat_history, response)
        return response
    
    # Retrieval
    try:
        corpora = await classify_llm(query)
        retrieved_chunks = await retrieve_chunks(query, corpora)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        # Continue with empty context but provide helpful response
        retrieved_chunks = []
    
    # Analyze conversation context for more intelligent responses
    context_analysis = analyze_conversation_context(chat_history, query)

    # Adjust prompt based on context
    enhanced_prompt = MASTER_PROMPT
    if context_analysis["emotional_state"] == "negative":
        enhanced_prompt += "\n\nلاحظ أن المتحدث يبدو قلقاً أو حزيناً. كن داعماً وعاطفياً في ردك."
    elif context_analysis["emotional_state"] == "confused":
        enhanced_prompt += "\n\nالمتحدث يبدو confused. شرح الأمور ببساطة ووضوح."
    elif context_analysis["repeated_questions"]:
        enhanced_prompt += "\n\nالمتحدث يسأل أسئلة متكررة. حاول تقديم إجابات مختلفة أو أعمق."

    if context_analysis["topics_discussed"]:
        enhanced_prompt += f"\n\nالمواضيع السابقة المطروحة: {', '.join(context_analysis['topics_discussed'])}"

    # Generate response
    try:
        context = "\n\n".join([c["chunk"] for c in retrieved_chunks]) if retrieved_chunks else ""
        messages = [
            {"role": "system", "content": enhanced_prompt.format(
                persona_preamble=PERSONA_PREAMBLE,
                chat_history=chat_history,
                context=context
            )},
            {"role": "user", "content": query}
        ]
        response_llm = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,  # Deterministic for final answers
        )
        final_answer = response_llm.choices[0].message.content.strip()

        # ----------- after persona_styler returns --------------
        # styled = persona_styler(... )  # existing call -> may return a string or JSON
        # We expect persona_styler to now return JSON with "styled_answer" and "styled_answer_diacritized"
        if isinstance(final_answer, dict):
            final_text = final_answer.get("styled_answer_diacritized") or final_answer.get("styled_answer")
        else:
            # old behavior: string. Apply diacritizer proactively.
            final_text = apply_diacritics(final_answer)

        # Now run validator (ensure validator receives both texts)
        validator_input = {
            "query": query,
            "answer": final_text,
            "context": context,
            "chat_history": chat_history
        }
        # For now, implement inline validator logic
        from utils.diacritizer import compute_diacritics_coverage
        coverage, per_word = compute_diacritics_coverage(final_text)
        validator_response = {
            "is_valid": True,
            "issues": [],
            "must_refine_query": False,
            "refine_instructions": "",
            "needs_persona_styling": False,
            "persona_score": 0.8,  # placeholder
            "diacritics_coverage": coverage,
            "diacritized_answer": ""
        }
        if coverage < 0.95:
            fixed = apply_diacritics(final_text)
            validator_response["diacritized_answer"] = fixed
            validator_response["is_valid"] = False
            validator_response["issues"].append("insufficient_diacritics")
        # detect missing sources
        if context and "مصادر" not in final_text and "مصدر" not in final_text:
            validator_response["issues"].append("missing_sources")
            validator_response["is_valid"] = False

        # If validator returns diacritized correction, use it
        if validator_response.get("diacritics_coverage", 0) < 0.95 and validator_response.get("diacritized_answer"):
            final_text = validator_response["diacritized_answer"]

        # Identify and append used sources
        used_sources = identify_used_sources(final_text, retrieved_chunks)
        if used_sources:
            sources_text = "\n\nمصادر:\n" + "\n".join([
                f"• {source['title']}" + (f" - {source['author']}" if source['author'] else "") +
                (f" ({source['date']})" if source['date'] else "")
                for source in used_sources[:3]  # Limit to 3 sources
            ])
            final_text += sources_text

        # Final output to user must be `final_text` (fully diacritized)
        response = final_text[:MAX_RESPONSE_LENGTH]
        set_cached_response(query, chat_history, response)
        return response
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        fallback_response = generate_fallback_response(query, "generation")
        set_cached_response(query, chat_history, fallback_response)
        return fallback_response

async def generate_tts(text: str) -> bytes:
    try:
        audio = elevenlabs.generate(
            text=text,
            voice="Rachel",  # Or appropriate voice
            model_id="eleven_monolingual_v1"
        )
        return audio
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, message_type: str):
    user_id = update.effective_user.id
    try:
        if message_type == "voice":
            try:
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
            except Exception as e:
                logger.error(f"Error in speech to text: {e}")
                await update.message.reply_text("عذراً، حدث خطأ في معالجة الصوت.")
                return
        elif message_type == "text":
            user_input = update.message.text
        else:
            await update.message.reply_text("الرجاء إرسال رسالة صوتية أو نصية فقط.")
            return
        
        history.add_message(user_id, "user", user_input)
        history.extract_and_store_name(user_id, user_input)
        chat_history = format_history_for_prompt(history.get_recent_history(user_id, max_messages=20))
        
        response = await generate_validated_response(user_id, user_input, chat_history)
        
        history.add_message(user_id, "system", response)
        await update.message.reply_text(response)
        
        # Optional TTS
        tts_audio = await generate_tts(response)
        if tts_audio:
            audio_io = BytesIO(tts_audio)
            audio_io.name = "response.mp3"
            await update.message.reply_voice(voice=audio_io)
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("عذراً، حدث خطأ. اللهم صلِّ على محمد وآل محمد.")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history.clear(user_id)
    await update.message.reply_text("تم مسح تاريخ المحادثة.")

# Main application
if __name__ == "__main__":
    nest_asyncio.apply()
    app = Application.builder().token(telegram_token).build()
    app.add_handler(CommandHandler("clear", clear_history))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_message, message_type="text")))
    app.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, partial(handle_message, message_type="voice")))
    
    print("Bot is running...")
    app.run_polling()