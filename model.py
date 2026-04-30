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

    def generate(self, query: str) -> str:
        """Retrieve context -> Prompt LLM -> Return safe, grounded response."""
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

        user_prompt = f"Context:\n{context}\n\nUser Question: {query}"
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
            return reply
        except Exception as e:
            return f"⚠️ LLM Error: {str(e)}"
