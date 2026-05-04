"use client";

import { useState } from "react";
import axios from "axios";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatInterface() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      // Prepare history for backend
      const historyStr = JSON.stringify(
        messages.filter((m) => m.role === "user" || m.role === "assistant")
      );

      const formData = new FormData();
      formData.append("query", input);
      formData.append("history", historyStr);

      const res = await axios.post("http://localhost:8000/chat", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const botMsg: Message = {
        role: "assistant",
        content: res.data.response,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: Could not reach backend." },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 h-[600px] flex flex-col">
      <div className="flex-1 overflow-y-auto mb-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-3 rounded-lg ${
              msg.role === "user"
                ? "bg-blue-500 text-white self-end"
                : "bg-gray-200 dark:bg-gray-700 self-start"
            }`}
          >
            <p className="whitespace-pre-wrap">{msg.content}</p>
          </div>
        ))}
        {isLoading && (
          <div className="text-gray-500 italic">Thinking...</div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a medical question..."
          className="flex-1 p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
        />
        <button
          type="submit"
          disabled={isLoading}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}

