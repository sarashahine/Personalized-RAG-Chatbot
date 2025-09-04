from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from io import BytesIO
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import openai
from sentence_transformers import SentenceTransformer
import faiss
import json
import sqlite3
import threading
from typing import Dict, List
import arabic_reshaper
from bidi.algorithm import get_display

# ------------------ Load APIs ------------------
load_dotenv()

class ChatHistoryManager:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_database()
        
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    user_id INTEGER PRIMARY KEY,
                    total_messages INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES chat_sessions (user_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_timestamp ON chat_messages (user_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_activity ON chat_sessions (last_activity)
            """)
    
    def add_message(self, user_id: int, role: str, content: str):
        with self.lock:            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO chat_messages (user_id, role, content) VALUES (?, ?, ?)",
                    (user_id, role, content)
                )
                conn.execute("""
                    INSERT INTO chat_sessions (user_id, total_messages)
                    VALUES (?, 1)
                    ON CONFLICT(user_id) DO UPDATE SET
                        total_messages = total_messages + 1
                """, (user_id,))
    
    def get_recent_history(self, user_id: int, max_messages: int = 20) -> List[Dict]:
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT role, content, timestamp FROM chat_messages 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (user_id, max_messages))
                
                messages = [{"role": row[0], "content": row[1], "timestamp": row[2]} 
                           for row in cursor.fetchall()]
                
                messages.reverse()
                return messages
    
    def clear_user_history(self, user_id: int):
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
                conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))

chat_manager = ChatHistoryManager()

elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")


model = SentenceTransformer("intfloat/multilingual-e5-large")
dim = model.get_sentence_embedding_dimension()

index = faiss.read_index("storage/index.faiss")
with open("storage/metadata.jsonl", "r", encoding="utf-8") as f:
    metadata = [json.loads(line) for line in f]

# ------------------ Helper Function ------------------
def generate_answer(query, retrieved_chunks, user_id, model_name="gpt-4o-mini"):
    context_parts = []
    for c in retrieved_chunks:
        if isinstance(c, dict):
            if "text" in c:
                context_parts.append(str(c["text"]))
            else:
                context_parts.append(str(c))
        else:
            context_parts.append(str(c))
    context = "\n\n".join(context_parts)

    history = chat_manager.get_recent_history(user_id, max_messages=10)
    
    formatted_history = []
    for msg in history:
        reshaped_text = arabic_reshaper.reshape(msg["content"])
        display_text = get_display(reshaped_text)
        print(display_text)

    for msg in history:
        formatted_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    system_message = (
        "أنت السيد هاشم صفي الدين، عالم ديني وقائد مقاوم."
        "عندما تتلقى تحية أو سؤالًا اجتماعيًا بسيطًا اكتفِ برد مختصر ومهذب يناسب الموقف، دون إضافة أي معلومات من قاعدة البيانات أو الحديث عن نفسك."
        "إذا كان السؤال متعلقًا بهويتك أو سيرتك الذاتية أو يطلب معلومات عنك، استخدم المعلومات التعريفية المتوفرة في السياق للإجابة بشكل مباشر وشخصي."
        "إذا كان السؤال متعلقًا بموضوع ديني أو تاريخي أو يطلب معلومة من السياق، استخدم أسلوبًا عربيًا فصيحًا يجمع بين الاحترام، العمق الديني، والسرد التاريخي كما هو ظاهر في النصوص المرفقة."
        # "ابدأ بالبسملة والصلاة على النبي وآله عند الحديث عن الأنبياء أو المواضيع الدينية."
        "التزم بالمنهجية العلمية، ووضح المفاهيم بدقة، وكن واقعيًا في الطرح، ولا تضف أي معلومة غير موجودة في السياق المقدم من قاعدة البيانات."
        "إذا لم تجد الجواب في النصوص، اعتذر بلباقة ووضوح."
        "احرص على ذكر الألقاب المناسبة للشخصيات الدينية، وراعِ الأدب في الحوار، وكن قريبًا من أسلوب الخطب والمحاضرات الدينية، مع سرد الأحداث والقصص بأسلوب مشوّق وموثق."
    )
    
    messages = [
        {
            "role": "system",
            "content": system_message,
        },
        *formatted_history,
        {
            "role": "user",
            "content": f"السياق:\n{context}\n\nالسؤال: {query}",
        },
    ]

    response = openai.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
    )
    answer = response.choices[0].message.content.strip()

    chat_manager.add_message(user_id, "user", query)
    chat_manager.add_message(user_id, "system", answer)

    return answer

async def reformulate_query(raw_query: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "# الهدف والدور"
                "- المساعد يمثل شخصية دينية وتاريخية محترمة، ويهدف إلى تقديم إجابات دقيقة وواضحة على أسئلة المستخدمين ضمن هذا السياق."

                "# التعليمات"
                "- يجب معالجة كل سؤال يُطرح من قبل المستخدم كما لو تم توجيهه إلى شخصية دينية وتاريخية محترمة."
                "- ينبغي تجهيز الرد بصياغة مفصلة، دقيقة، وشاملة من حيث الفهم."
                "- من المهم الحفاظ على مضمون السؤال وطريقة عرضه من قبل المستخدم في الإجابة."

                "## الإرشادات الفرعية"
                "- يجب أن تكون الإجابة محترمة وخالية من أي تحامل أو إساءة."
                "- استخدم لغة عربية فصحى واضحة وسلسة."
                "- احرص على تغطية جميع جوانب السؤال وعدم الاقتصار على ملخص موجز."

                "# الخطوات الأولية"
                "- ابدأ بإعداد قائمة مختصرة (3-7 نقاط) بالمراحل أو الخطوات المفاهيمية التي ستتبعها للإجابة على السؤال، ابتعد عن التفاصيل التنفيذية."

                "# السياق"
                "- الأسئلة تأتي من مستخدمين يوجهونها إلى مساعد افتراضي يجسد شخصية دينية وتاريخية."
                "- الردود يجب أن تعكس وقار هذه الشخصية وتحترم مصادر المعرفة الدينية والتاريخية."
                "- الأسئلة الدينية أو التاريخية المعقدة ضمن النطاق، والإجابات الأدبية أو السطحية خارجه."

                "# خطة العمل والتحقق"
                "- راجع نص السؤال للتأكد من استيفاء جميع العناصر المذكورة في التعليمات."
                "- تحقق أن الإجابة دقيقة، واضحة، وتحافظ على أسلوب المستخدم."
                "- تأكد من عدم وجود أي محتوى مسيء أو غير دقيق تاريخياً أو دينياً."


                # مستوى التفصيل
                "- الاهتمام بالدقة والفهم العميق للمحتوى."

                # شروط الانتهاء
                "- التوقف بعد التأكد من تحقيق جميع الشروط السابقة."
                "- إذا لم يكن بالإمكان تقديم إجابة ضمن هذه الشروط، يجب الاعتذار بشكل محترم ولبق."
            )},
            {"role": "user", "content": raw_query},
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


# ------------------ Telegram Handlers ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    welcome_msg = "👋 مرحباً! أنا السيد هاشم صفي الدين. أرسل لي نصاً أو رسالة صوتية وسأرد عليك."
    
    await update.message.reply_text(welcome_msg)

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id
    chat_manager.clear_user_history(user_id)
    await update.message.reply_text("🗑️ تم مسح تاريخ المحادثة بنجاح.")

async def history_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = update.effective_user.id


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw_query = update.message.text
    query = await reformulate_query(raw_query)
    user_id = update.effective_user.id
    
    q_emb = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    k = 4
    D, I = index.search(q_emb, k)
    retrieved_chunks = [metadata[idx] for idx in I[0]]

    answer = generate_answer(query, retrieved_chunks, user_id)

    await update.message.reply_text(answer)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    voice_file = await update.message.voice.get_file()
    audio_bytes = BytesIO()
    await voice_file.download_to_memory(out=audio_bytes)
    audio_bytes.seek(0)

    try:
        transcription = elevenlabs.speech_to_text.convert(
            file=audio_bytes,
            model_id="scribe_v1",
            tag_audio_events=True,
            language_code="ara",
            diarize=True,
        )
        raw_query = transcription.text
        query = await reformulate_query(raw_query)
    except Exception as e:
        query = f"[Transcription failed] {e}"

    user_id = update.effective_user.id

    q_emb = model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    k = 3
    D, I = index.search(q_emb, k)
    retrieved_chunks = [metadata[idx] for idx in I[0]]

    answer = generate_answer(query, retrieved_chunks, user_id)
    print("Answer:", answer)

    await update.message.reply_text(answer)

# ------------------ Build Bot ------------------
app = Application.builder().token("8440954235:AAFf1SA4l0aTHMrQwErX3w7syqKZdWdWACU").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("clear", clear_history))
app.add_handler(CommandHandler("stats", history_stats))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
app.add_handler(MessageHandler(filters.VOICE, handle_voice))

# ------------------ Run Bot ------------------
if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
    print("Bot has stopped.")