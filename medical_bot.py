from model import MedicalRAG

if __name__ == "__main__":
    print("🏥 Initializing Medical RAG Chatbot...")
    print("⏳ Loading MedQuAD dataset & FAISS index...")

    # Initialize RAG pipeline
    rag_bot = MedicalRAG()

    print("✅ Ready! Type your health question (or 'quit' to exit).\n")

    history = []  # Track conversation history

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAssistant: Take care! Always consult a licensed healthcare professional for medical advice. Goodbye! 👋")
            break

        # Generate Response
        response = rag_bot.generate(user_input, history=history)
        print(f"\nAssistant: {response['response']}\n")
        
        # Update history (limit to last 6 messages to control context size)
        if response['accepted']:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            if len(history) > 6:
                history = history[-6:]
