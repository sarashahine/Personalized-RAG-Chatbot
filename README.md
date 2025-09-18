# Personalized RAG Chatbot - Sayed Hashem Safieddine

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that embodies the speaking style and personality of Sayed Hashem Safieddine, a prominent Lebanese Shia religious scholar and political leader.

## 🚀 Features

- **Authentic Persona Emulation**: Mimics the rhetorical style, vocabulary, and discourse patterns of Sayed Hashem Safieddine
- **Arabic Language Support**: Full support for Arabic text processing and generation
- **Multi-stage LLM Pipeline**: Advanced processing with planner, reformulator, classifier, retriever, generator, validator, and styler
- **Telegram Bot Integration**: Real-time conversational interface via Telegram
- **Voice Input/Output**: Speech-to-text and text-to-speech capabilities using ElevenLabs
- **FAISS Vector Search**: Efficient semantic search through religious texts and speeches
- **Redis Caching**: Response caching and user session management
- **Multi-part Query Handling**: Intelligent processing of complex, multi-part questions

## 📁 Project Structure

```
Personalized-RAG-Chatbot/
├── src/                          # Source code
│   ├── bot.py                   # Main Telegram bot implementation
│   ├── prompts.py               # LLM prompt templates
│   ├── shared_redis.py          # Redis utilities for caching and sessions
│   └── main.py                  # Alternative main script
├── config/                      # Configuration files
│   ├── character.json           # Persona definition and rhetorical guidelines
│   ├── personality_instructions.json  # Detailed linguistic analysis
│   └── character_aside.json     # Alternative persona configuration
├── data/                        # Data files and indexes
│   ├── chunks.json              # Text chunks for RAG
│   ├── chunks/                  # Additional chunk files
│   └── storage/
│       ├── chunks_metadata.json # FAISS index metadata
│       └── openai_index.faiss   # FAISS vector index
├── notebooks/                   # Jupyter notebooks for development
│   ├── create_embeddings.ipynb  # Embedding creation pipeline
│   ├── full_code.ipynb          # Complete implementation notebook
│   ├── stt.ipynb               # Speech-to-text experiments
│   ├── tts.ipynb               # Text-to-speech experiments
│   ├── tomp3.ipynb             # Audio processing
│   ├── add_new_embed.ipynb     # Adding new embeddings
│   ├── character.ipynb         # Character analysis
│   ├── chunks.ipynb            # Chunk processing
│   ├── embedding.ipynb         # Embedding experiments
│   ├── embed_prompt.ipynb      # Prompt engineering
│   ├── full.ipynb              # Full pipeline notebook
│   ├── stt_aside.ipynb         # Alternative STT notebook
│   ├── tomp3_aside.ipynb       # Alternative audio processing
│   └── tts_aside.ipynb         # Alternative TTS notebook
├── audio/                      # Audio files and samples
│   ├── output.mp3              # Generated audio samples
│   ├── output2.mp3
│   ├── sayed_clips/            # Original audio clips
│   └── voices/                 # Voice samples
├── docs/                       # Documentation and transcripts
│   ├── transcription/          # Audio transcription files
│   └── report.docx             # Project documentation
├── tests/                      # Test files
│   └── test_validation.py      # Validation test suite
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── .env                        # Environment variables (not in git)
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarashahine/Personalized-RAG-Chatbot.git
   cd Personalized-RAG-Chatbot
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   REDIS_URL=redis://localhost:6379  # Optional, defaults to local Redis
   ```

## 🚀 Usage

### Running the Telegram Bot

```bash
python src/bot.py
```

The bot will start polling for messages on Telegram. Users can interact with it by sending text or voice messages.

### Alternative Main Script

```bash
python src/main.py
```

### Testing the Pipeline

```bash
python tests/test_validation.py
```

## 🧠 Architecture

The chatbot uses a sophisticated multi-stage LLM pipeline:

1. **Planner**: Analyzes user input and determines the response strategy
2. **Reformulator**: Rephrases queries for optimal retrieval
3. **Classifier**: Categorizes queries by content type (tafsir, hadith, speeches, etc.)
4. **Retriever**: Searches FAISS vector index for relevant text chunks
5. **Generator**: Creates initial response using retrieved context
6. **Validator**: Ensures response accuracy and persona compliance
7. **Styler**: Transforms response to match Sayed Hashem's authentic voice

### Key Components

- **Persona Emulation**: Detailed analysis of linguistic patterns, vocabulary, and rhetorical devices
- **RAG System**: Combines retrieval from religious texts with generative AI
- **Voice Integration**: ElevenLabs for high-quality Arabic speech synthesis
- **Caching**: Redis-based response caching for improved performance
- **Session Management**: User-specific conversation history and preferences

## 🎯 Persona Characteristics

The bot emulates Sayed Hashem Safieddine's speaking style with:

- **Formal Religious Register**: Scholarly Arabic with Lebanese colloquial touches
- **Rhetorical Structure**: Varied transitions avoiding repetitive patterns
- **Moral Emphasis**: Focus on virtues like عفة (chastity), ورع (piety), قناعة (contentment)
- **Islamic References**: Citations from Quran, Hadith, and Ahl al-Bayt
- **Didactic Tone**: Uplifting, admonitory, and compassionate delivery

## 📊 Data Sources

The RAG system is trained on:
- Quranic exegesis (tafsir)
- Prophetic traditions (hadith)
- Religious speeches and lectures
- Historical Islamic texts
- Sayed Hashem's personal writings and speeches

## 🔧 Configuration

### Character Configuration
Modify `config/character.json` to adjust:
- Lexical features and vocabulary
- Syntactic patterns
- Rhetorical scaffolds
- Greeting and closing templates

### Personality Instructions
`config/personality_instructions.json` contains detailed linguistic analysis for fine-tuning the persona.

### Prompts
Customize LLM behavior in `src/prompts.py`:
- System prompts for each pipeline stage
- Response formatting guidelines
- Validation criteria

## 🧪 Development

### Running Notebooks
Use Jupyter notebooks in the `notebooks/` directory for:
- Experimenting with embeddings
- Testing STT/TTS functionality
- Analyzing character data
- Developing new features

### Adding New Content
1. Process new text/audio content
2. Generate embeddings using `create_embeddings.ipynb`
3. Update FAISS index
4. Test with validation scripts

## 📈 Performance Optimization

- **Model Selection**: Uses GPT-4o-mini for speed and cost efficiency
- **Caching**: Redis-based response caching reduces API calls
- **Vector Search**: FAISS enables fast semantic retrieval
- **Async Processing**: Concurrent LLM calls for improved responsiveness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect intellectual property rights and religious sensitivities.

## 🙏 Acknowledgments

- Sayed Hashem Safieddine for his inspirational teachings
- OpenAI for LLM capabilities
- ElevenLabs for voice synthesis
- FAISS for vector search
- Telegram for bot platform

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Note**: This chatbot is designed to respectfully emulate a religious figure's communication style. It should be used responsibly and in accordance with Islamic principles and ethical AI guidelines.