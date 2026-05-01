import os
import json
import pandas as pd
import numpy as np
import faiss
import ollama

class MedicalRAG:
    def __init__(self, kb_path="knowledge/medquad.csv", index_path="knowledge/faiss.index", chunks_path="knowledge/chunks.json", embed_model="nomic-embed-text:v1.5", chat_model="phi3.5:3.8b", k=3):
        self.kb_path = kb_path
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.k = k
        self.index = None
        self.chunks = []
        self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self._load_index()
        else:
            self._build_index()

    def _build_index(self):
        print("📖 Loading MedQuAD dataset...")
        df = pd.read_csv(self.kb_path)
        df["chunk"] = (
            "Question: " + df["question"].fillna("") + "\n"
            "Answer: " + df["answer"].fillna("") + "\n"
            "Source: " + df["source"].fillna("Unknown") + "\n"
            "Focus Area: " + df["focus_area"].fillna("General")
        )
        all_chunks = df["chunk"].tolist()

        print(f"🧠 Embedding {len(all_chunks)} chunks (one-by-one)...")
        valid_chunks = []
        valid_embeddings = []

        for i, chunk in enumerate(all_chunks):
            try:
                res = ollama.embed(model=self.embed_model, input=chunk)
                valid_embeddings.append(res["embeddings"][0])
                valid_chunks.append(chunk)
                print(f"  Progress: {len(valid_embeddings)}/{len(all_chunks)}", end="\r")
            except Exception as e:
                print(f"\n  ⚠️ Skipping chunk {i+1} due to error: {e}")
                continue

        # Sync chunks list with successfully embedded ones
        self.chunks = valid_chunks
        if not valid_embeddings:
            raise RuntimeError("No chunks were successfully embedded. Check Ollama server & model.")

        print(f"\n✅ Successfully embedded {len(self.chunks)} chunks. Building index...")
        embeddings = np.array(valid_embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        print("✅ FAISS index built and saved.")

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"📚 Loaded index with {len(self.chunks)} chunks.")

    def _is_medical_topic(self, text: str) -> bool:
        """Check if input is health-related using keyword matching."""
        medical_keywords = [
            "headache", "fever", "cold", "cough", "pain", "symptom", "health",
            "diet", "nutrition", "anxiety", "stress", "sleep", "vitamin",
            "doctor", "hospital", "medicine", "drug", "disease", "treatment",
            "blood", "heart", "lung", "skin", "bone", "virus", "bacteria"
        ]
        return any(kw in text.lower() for kw in medical_keywords)

    def _detect_urgency(self, text: str) -> bool:
        """Detect emergency-level keywords requiring immediate routing."""
        urgent_keywords = [
            "emergency", "chest pain", "can't breathe", "fainting", "stroke",
            "bleeding heavily", "suicidal", "unconscious", "severe allergic reaction"
        ]
        return any(kw in text.lower() for kw in urgent_keywords)

    def retrieve(self, query: str) -> list[str]:
        """Semantic search: returns top-k relevant chunks."""
        try:
            res = ollama.embed(model=self.embed_model, input=query)
            q_emb = np.array(res["embeddings"][0]).astype("float32").reshape(1, -1)
            _, indices = self.index.search(q_emb, self.k)
            return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        except Exception as e:
            print(f"⚠️ Retrieval error: {e}")
            return []

    def generate(self, query: str, history: list[dict] = None) -> dict:
        """Full pipeline: safety checks → retrieval → LLM generation → safe response."""
        
        # Domain Restriction (inside model)
        if not self._is_medical_topic(query):
            return {
                "response": "I'm designed to assist with health and wellness topics only. Please ask about symptoms, conditions, nutrition, or general medical guidance.\n⚠️ Disclaimer: I am an AI assistant, not a licensed healthcare professional. This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.",
                "accepted": False
            }

        # Urgency Handling (prepend emergency warning if needed)
        prefix = ""
        if self._detect_urgency(query):
            prefix = "If you are experiencing a medical emergency, please call your local emergency number (e.g., 911) or go to the nearest hospital immediately.\n\n"

        # Retrieve context (only for medical queries)
        context_chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

        system_prompt = """
        ROLE: Basic Medical Information Assistant
        RULES:
        - Answer ONLY using the provided context.
        - If the context is missing, irrelevant, or doesn't answer the question, say:
          "I don't have verified information on that in my medical database. Please consult a licensed healthcare provider."
        - NEVER diagnose, prescribe, or recommend specific medications.
        - Keep responses simple, factual, and calm.
        - ALWAYS end with this exact disclaimer:
          "⚠️ Disclaimer: I am an AI assistant, not a licensed healthcare professional. This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment."
        """

        # Maintaining conversation history
        conversation = ""
        if history:
            for msg in history[-4:]:  # Keep last 4 messages to avoid context overflow
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation += f"{role}: {msg['content']}\n"
        
        user_prompt = f"Context:\n{context}\n\nConversation History:\n{conversation}User Question: {query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            res = ollama.chat(model=self.chat_model, messages=messages, stream=False)
            reply = res["message"]["content"].strip()
            # Safety fallback: guarantee disclaimer
            if "Disclaimer" not in reply and "disclaimer" not in reply.lower():
                reply += "\n\n⚠️ Disclaimer: I am an AI assistant, not a licensed healthcare professional. This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment."
            
            # Prepend urgency warning if needed
            # In model.py - generate() return:
            return {
                "response": prefix + reply,
                "accepted": True  # or False if domain-rejected
            }
            
        except Exception as e:
            return {
                "response": f"⚠️ LLM Error: {str(e)}",
                "accepted": False
            }
