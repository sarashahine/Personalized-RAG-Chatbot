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
    PLANNER_PROMPT,
    REFORMULATOR_PROMPT,
    CLASSIFIER_PROMPT,
    ANSWER_GENERATION_PROMPT,
    VALIDATOR_PROMPT,
    PERSONA_STYLE_PROMPT,
)

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

async def get_cached_response(query: str, user_id: int) -> str:
    cache_key = f"{user_id}:{query.lower().strip()}"
    return response_cache.get(cache_key)

def set_cached_response(query: str, user_id: int, response: str):
    cache_key = f"{user_id}:{query.lower().strip()}"
    response_cache[cache_key] = response

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

async def planner_llm(user_message: str, chat_history: str) -> Dict:
    try:
        messages = [
            {"role": "system", "content": PLANNER_PROMPT.format(persona_preamble=PERSONA_PREAMBLE)},
            {"role": "user", "content": f"User message: {user_message}\nChat history: {chat_history}"}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        result = json.loads(response.choices[0].message.content.strip())
        # Validate JSON structure
        if not isinstance(result, dict):
            raise ValueError("Invalid JSON structure")
        # Add defaults for missing keys
        if "needs_research" not in result:
            result["needs_research"] = True
        if "special_handling" not in result:
            result["special_handling"] = None
        if "strategy" not in result:
            result["strategy"] = "general"
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from planner LLM: {e}")
        return {"strategy": "general", "needs_research": True, "special_handling": None}
    except Exception as e:
        logger.error(f"Error in planner_llm: {e}")
        return {"strategy": "general", "needs_research": True, "special_handling": None}

async def reformulator_llm(query: str, chat_history: str) -> str:
    try:
        messages = [
            {"role": "system", "content": REFORMULATOR_PROMPT.format(persona_preamble=PERSONA_PREAMBLE)},
            {"role": "user", "content": f"Query: {query}\nChat history: {chat_history}"}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in reformulator_llm: {e}")
        return query  # Return original query if error

async def classify_llm(query: str) -> Dict:
    # Deterministic classification for simplicity - make it async for consistency
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
    
    special_case = any(word in query_lower for word in ["تحية", "من أنت", "رقم", "حساس"])
    
    return {
        "corpora": corpora or ["general"],
        "is_special_case": special_case
    }

async def retrieve_chunks(query: str, corpora: List[str], k: int = 5) -> List[Dict]:
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

async def generate_answer_llm(query: str, retrieved_chunks: List[Dict], chat_history: str) -> str:
    try:
        context = "\n\n".join([c["chunk"] for c in retrieved_chunks]) if retrieved_chunks else ""
        messages = [
            {"role": "system", "content": ANSWER_GENERATION_PROMPT.format(
                persona_preamble=PERSONA_PREAMBLE,
                chat_history=chat_history,
                context=context
            )},
            {"role": "user", "content": query}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in generate_answer_llm: {e}")
        return "عذراً، حدث خطأ في توليد الإجابة."

async def validate_llm(query: str, answer: str, retrieved_chunks: List[Dict], chat_history: str) -> Dict:
    try:
        context = "\n\n".join([c["chunk"] for c in retrieved_chunks]) if retrieved_chunks else ""
        messages = [
            {"role": "system", "content": VALIDATOR_PROMPT.format(
                persona_preamble=PERSONA_PREAMBLE,
                chat_history=chat_history,
                context=context
            )},
            {"role": "user", "content": f"Query: {query}\nAnswer: {answer}"}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
        )
        result = json.loads(response.choices[0].message.content.strip())
        # Validate JSON structure
        if not isinstance(result, dict):
            raise ValueError("Invalid JSON structure")
        # Add defaults for missing keys
        if "is_valid" not in result:
            result["is_valid"] = True
        if "issues" not in result:
            result["issues"] = []
        if "must_refine_query" not in result:
            result["must_refine_query"] = False
        if "refine_instructions" not in result:
            result["refine_instructions"] = ""
        if "needs_persona_styling" not in result:
            result["needs_persona_styling"] = True
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from validator LLM: {e}")
        return {"is_valid": True, "issues": [], "must_refine_query": False, "refine_instructions": "", "needs_persona_styling": True}
    except Exception as e:
        logger.error(f"Error in validate_llm: {e}")
        return {"is_valid": True, "issues": [], "must_refine_query": False, "refine_instructions": "", "needs_persona_styling": True}

async def persona_style_llm(answer: str, chat_history: str) -> str:
    try:
        messages = [
            {"role": "system", "content": PERSONA_STYLE_PROMPT.format(
                persona_preamble=PERSONA_PREAMBLE,
                chat_history=chat_history
            )},
            {"role": "user", "content": answer}
        ]
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in persona_style_llm: {e}")
        return answer  # Return original answer if error

async def generate_validated_response(user_id: int, query: str, chat_history: str) -> str:
    iteration = 0
    current_query = query
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        # Step 1: Planner
        plan = await planner_llm(current_query, chat_history)
        
        # Check for name-related queries
        query_lower = current_query.lower()
        if "shu esme" in query_lower or "what is my name" in query_lower or "اسمي" in query_lower:
            user_profile = history.get_user_profile(user_id)
            name = user_profile.get("name")
            if name:
                return f"أعوذ بالله من شر الشيطان الرجيم. بسم الله الرحمن الرحيم.\n\nاسمك {name}، يا {name}. كيف يمكنني خدمتك اليوم؟"[:MAX_RESPONSE_LENGTH]
        
        # Handle special cases
        if plan.get("special_handling"):
            if plan["special_handling"] == "greeting":
                user_profile = history.get_user_profile(user_id)
                name = user_profile.get("name", "")
                greeting = "السلام عليكم ورحمة الله وبركاته."
                if name:
                    greeting += f" أهلاً وسهلاً بك، يا {name}."
                else:
                    greeting += " أهلاً وسهلاً بك."
                greeting += " كيف يمكنني خدمتك اليوم؟"
                return greeting[:MAX_RESPONSE_LENGTH]
            elif plan["special_handling"] == "identity":
                return "أعوذ بالله من شر الشيطان الرجيم. بسم الله الرحمن الرحيم.\n\nأنا هاشم صفيّ الدين، خادم لأهل البيت سلام الله عليهم. أقدم لكم النصائح والمعلومات المستندة إلى القرآن الكريم والحديث الشريف وتراث أهل البيت. كيف يمكنني خدمتكم اليوم؟"[:MAX_RESPONSE_LENGTH]
            # Add other special cases here
        
        # Step 2: Reformulator
        reformulated_query = await reformulator_llm(current_query, chat_history)
        
        # Step 3: Classifier (only if research is needed)
        if plan.get("needs_research", True):
            classification = classify_llm(reformulated_query)
            corpora = classification["corpora"]
        else:
            corpora = []
        
        # Step 4: Retrieval (conditional)
        if plan.get("needs_research", True):
            retrieved_chunks = await retrieve_chunks(reformulated_query, corpora)
        else:
            retrieved_chunks = []
        
        # Step 5: Answer Generation
        draft_answer = await generate_answer_llm(reformulated_query, retrieved_chunks, chat_history)
        
        # Step 6: Validator
        validation = await validate_llm(reformulated_query, draft_answer, retrieved_chunks, chat_history)
        
        if validation["is_valid"]:
            # Step 7: Persona Style Pass (always apply for character priority)
            final_answer = await persona_style_llm(draft_answer, chat_history)
            return final_answer[:MAX_RESPONSE_LENGTH]
        
        if validation.get("must_refine_query", False):
            current_query = validation.get("refine_instructions", current_query)
            continue
        
        # If invalid but no refinement needed, return anyway
        return f"{draft_answer}\n\nملاحظات: {', '.join(validation['issues'])}"[:MAX_RESPONSE_LENGTH]
    
    # Fallback
    return "عذراً، حدث خطأ في معالجة الاستعلام. يرجى المحاولة مرة أخرى."

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