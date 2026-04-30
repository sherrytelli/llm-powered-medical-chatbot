# 🏥 Medical RAG Chatbot

A responsible, domain-specific medical information assistant powered by **Retrieval-Augmented Generation (RAG)**, **Ollama**, and the **MedQuAD dataset**. Designed for educational purposes with safety-first principles.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Ollama](https://img.shields.io/badge/ollama-local%20LLM-orange)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

- 🔍 **RAG-Powered Answers**: Grounds responses in verified MedQuAD Q&A pairs to reduce hallucinations
- 🛡️ **Multi-Layer Safety**: Domain filtering + urgency detection + strict system prompt + programmatic disclaimer fallback
- 💬 **Context-Aware Chat**: Maintains short-term conversation history for natural follow-up questions
- 🔐 **100% Local**: Runs entirely offline with Ollama — no data leaves your machine
- ⚙️ **Highly Configurable**: Swap models, adjust retrieval parameters, or point to custom datasets via constructor

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Python 3.11 or higher
- [Ollama](https://ollama.com) installed and running locally

### 2️⃣ Clone & Setup Virtual Environment
```bash
# Clone project
git clone https://github.com/sherrytelli/llm-powered-medical-chatbot.git

# Navigate to project directory
cd llm-powered-medical-chatbot

# Create and activate virtual environment
python -m venv venv

# Activate:
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install pandas numpy faiss-cpu ollama
```

### 4️⃣ Pull Required Ollama Models
```bash
# Default models (can be overridden via constructor)
ollama pull phi3.5:3.8b          # Chat model (~2.3GB)

ollama pull nomic-embed-text:v1.5  # Embedding model (~270MB)

# Verify models are ready:
ollama list
```

### 5️⃣ Run the Chatbot
```bash
python medical_bot.py
```

> 💡 **First run note**: The project includes a **pre-computed FAISS index** (`knowledge/faiss.index`) and chunk cache (`knowledge/chunks.json`) so you can start chatting immediately. Index building from scratch (~15k MedQuAD rows) takes 20 or more minutes depending on hardware.

---

## ⚙️ Configuration: Parameterized Constructor

The `MedicalRAG` class in `model.py` uses a flexible, parameterized constructor to let you customize every aspect of the pipeline:

```python
rag_bot = MedicalRAG(
    kb_path="knowledge/medquad.csv",      # Path to your MedQuAD CSV
    index_path="knowledge/faiss.index",   # Where to save/load FAISS index
    chunks_path="knowledge/chunks.json",  # Where to save/load text chunks
    embed_model="nomic-embed-text:v1.5",  # Ollama embedding model name
    chat_model="phi3.5:3.8b",            # Ollama chat model name
    k=3                                   # Top-K chunks to retrieve per query
)
```

### 🔧 Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kb_path` | `str` | `"knowledge/medquad.csv"` | Path to the MedQuAD CSV dataset |
| `index_path` | `str` | `"knowledge/faiss.index"` | Path to save/load the FAISS vector index |
| `chunks_path` | `str` | `"knowledge/chunks.json"` | Path to save/load pre-processed text chunks |
| `embed_model` | `str` | `"nomic-embed-text:v1.5"` | Ollama model for generating embeddings. Try: `all-minilm`, `bge-m3` |
| `chat_model` | `str` | `"phi3.5:3.8b"` | Ollama model for response generation. Try: `llama3.2`, `gemma2:2b`, `mistral` |
| `k` | `int` | `3` | Number of top-relevant chunks to retrieve per query. Higher = more context but slower |

### 🔄 Swap Models Example
Want to use Llama 3.2 instead of Phi-3.5?
```bash
# Pull the new model
ollama pull llama3.2

# pass args to constructor:
rag_bot = MedicalRAG(
    chat_model="llama3.2",
    embed_model="nomic-embed-text:v1.5"
)
```

---

## 📁 Project Structure

```
llm-powered-medical-chatbot/
├── medical_bot.py          # Entry point: chat loop, safety filters, UX
├── model.py                # MedicalRAG class: RAG pipeline, Ollama integration
├── knowledge/
│   ├── faiss.index         # ✅ Pre-computed vector index (ready to use)
│   └── chunks.json         # ✅ Pre-processed text chunks (aligned with index)
├── README.md               # This file
└── LICENSE                 # MIT License
```

---

## 🗃️ Pre-Computed Index (Skip the Wait!)

Building the FAISS index from scratch requires embedding ~15,000 MedQuAD rows one-by-one, which can take **20-40 minutes** on typical hardware.

✅ **Good news**: This repo includes pre-computed artifacts:
- `knowledge/faiss.index` — FAISS vector index (ready for semantic search)
- `knowledge/chunks.json` — Aligned text chunks for retrieval

👉 **Just run `python medical_bot.py`** and start chatting immediately. The system will auto-detect and load these files.

### 🔁 Rebuild Index (Optional)
If you modify the dataset or want to use a different embedding model:
```bash
# Delete pre-computed files
rm knowledge/faiss.index knowledge/chunks.json

# Run the bot — it will auto-rebuild
python medical_bot.py
```

> ⚠️ Rebuilding requires Ollama server running and ~4-6GB free RAM.

---

## 📚 Dataset Citation

This project uses the **MedQuAD (Medical Question Answering Dataset)**:

**Source**: [https://www.kaggle.com/datasets/gpreda/medquad](https://www.kaggle.com/datasets/gpreda/medquad)  
**License**: Dataset is for research/educational use. Verify usage terms on Kaggle.

---

## 🛡️ Safety & Responsible AI Design

This chatbot implements **defense-in-depth safety**:

| Layer | Implementation | Purpose |
|-------|---------------|---------|
| **Input Filter** | `is_medical_topic()` keyword check | Blocks off-topic queries early |
| **Urgency Detector** | `detect_urgency()` emergency keyword scan | Routes critical symptoms to emergency guidance |
| **RAG Grounding** | Answers only from retrieved MedQuAD context | Prevents hallucinations & unsupported claims |
| **System Prompt** | Strict role + rules + mandatory disclaimer | Guides LLM behavior at inference time |
| **Code Fallback** | Programmatic disclaimer append | Guarantees compliance even if prompt is ignored |
| **No Diagnosis/Prescription** | Explicit rule in prompt + logic | Prevents harmful medical advice |

> ⚠️ **Disclaimer**: This tool is for **educational purposes only**. It does not replace professional medical advice, diagnosis, or treatment. Always consult a licensed healthcare provider for personal health concerns.

---

## 🧪 Testing & Demo Queries

Try these to validate functionality:

```text
✅ Grounded answer:
You: What causes tension headaches?
Assistant: [Retrieves MedQuAD answer about stress, posture, dehydration] + disclaimer

✅ Follow-up with context:
You: What about children?
Assistant: [References pediatric context from history + new retrieval] + disclaimer

✅ Safety refusal:
You: Can you prescribe amoxicillin for my fever?
Assistant: I cannot prescribe medications. Please consult a licensed healthcare provider. + disclaimer

✅ Domain filter:
You: What's the best Python web framework?
Assistant: I'm designed to assist with health topics only... + disclaimer

✅ Emergency routing:
You: I have chest pain and can't breathe
Assistant: 🚨 If you are experiencing a medical emergency, please call 911... + fallback info
```

---

## 🤝 Contributing

Contributions welcome! To propose changes:
1. Fork the repo
2. Create a feature branch (`git checkout -b feat/your-idea`)
3. Commit changes (`git commit -m 'Add: your improvement'`)
4. Push and open a Pull Request

Please ensure:
- Safety constraints remain intact
- New dependencies are justified and documented
- Code follows existing style (PEP 8)

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---