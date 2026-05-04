# Medical RAG Chatbot - Full Stack Branch

This is the full-stack version of the Medical RAG Chatbot with a FastAPI backend and Next.js frontend.

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm
- [Ollama](https://ollama.com) installed and running locally

## Setup

### 1. Clone the Repository and Branch
```bash
# Clone the repository
git clone https://github.com/sherrytelli/llm-powered-medical-chatbot.git

# Navigate to the project
cd llm-powered-medical-chatbot

# Checkout the full-stack branch
git checkout -b full-stack origin/full-stack
```

### 2. Backend (FastAPI)

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
fastapi run dev
```

### 3. Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

## Pull Required Ollama Models

```bash
ollama pull phi3.5:3.8b
ollama pull nomic-embed-text:v1.5
```

## Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
