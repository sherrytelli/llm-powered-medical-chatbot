from model import MedicalRAG

def detect_urgency(text: str) -> bool:
    urgent_keywords = [
        "emergency", "chest pain", "can't breathe", "fainting", "stroke",
        "bleeding heavily", "suicidal", "unconscious", "severe allergic reaction"
    ]
    return any(kw in text.lower() for kw in urgent_keywords)

def is_medical_topic(text: str) -> bool:
    medical_keywords = [
        "headache", "fever", "cold", "cough", "pain", "symptom", "health",
        "diet", "nutrition", "anxiety", "stress", "sleep", "vitamin",
        "doctor", "hospital", "medicine", "drug", "disease", "treatment",
        "blood", "heart", "lung", "skin", "bone", "virus", "bacteria"
    ]
    return any(kw in text.lower() for kw in medical_keywords)

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

        # Domain Restriction
        if not is_medical_topic(user_input):
            print("\nAssistant: I'm designed to assist with health and wellness topics only. Please ask about symptoms, conditions, nutrition, or general medical guidance.")
            print("⚠️ Disclaimer: I am an AI assistant, not a licensed healthcare professional. This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment.\n")
            continue

        # Urgency Check
        if detect_urgency(user_input):
            print("\nAssistant: 🚨 If you are experiencing a medical emergency, please call your local emergency number (e.g., 911) or go to the nearest hospital immediately.")

        # Generate Response
        response = rag_bot.generate(user_input, history=history)
        print(f"\nAssistant: {response}\n")
        
        # Update history (limit to last 6 messages to control context size)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 6:
            history = history[-6:]
