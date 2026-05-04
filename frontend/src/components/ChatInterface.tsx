"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm your Medical RAG Assistant. I can help answer health-related questions based on verified medical information. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const historyStr = JSON.stringify(
        messages.filter((m) => m.role === "user" || m.role === "assistant")
      );

      const formData = new FormData();
      formData.append("query", input.trim());
      formData.append("history", historyStr);

      const res = await axios.post("http://localhost:8000/chat", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const botMessage: Message = {
        role: "assistant",
        content: res.data.response,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I apologize, but I'm having trouble connecting to the server. Please try again later.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const isUserMessage = (role: string) => role === "user";

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] max-w-5xl mx-auto px-4 py-6">
      {/* Header */}
      <div className="text-center mb-6 animate-slide-up">
        <div className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-full text-sm text-blue-700 dark:text-blue-300 mb-4">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span>AI Assistant Online</span>
        </div>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
          Medical RAG Assistant
        </h1>
        <p className="text-gray-600 dark:text-gray-300">Powered by Retrieval-Augmented Generation</p>
      </div>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 px-2">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${isUserMessage(message.role) ? "justify-end" : "justify-start"} animate-slide-up`}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex items-start space-x-3 max-w-[85%]">
              {!isUserMessage(message.role) && (
                <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
              )}
              <div className={`flex-1 ${isUserMessage(message.role) ? "items-end" : "items-start"}`}>
                <div
                  className={`chat-bubble ${
                    isUserMessage(message.role)
                      ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white user"
                      : "bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 assistant shadow-sm border border-gray-200 dark:border-gray-700"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                </div>
                <div className={`text-xs text-gray-500 dark:text-gray-400 mt-1 ${isUserMessage(message.role) ? "text-right" : "text-left"}`}>
                  {formatTime(message.timestamp)}
                  {!isUserMessage(message.role) && " • AI"}
                </div>
              </div>
              {isUserMessage(message.role) && (
                <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
                  <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start animate-slide-up">
            <div className="flex items-center space-x-3">
              <div className="flex-shrink-0 w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="bg-white dark:bg-gray-800 rounded-2xl rounded-bl-md p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-sm text-gray-600 dark:text-gray-300">Thinking...</span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="glass rounded-2xl p-4 shadow-lg">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex space-x-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a medical question..."
                className="w-full px-5 py-3 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900 dark:text-white placeholder-gray-500 transition-all duration-200"
                disabled={isLoading}
              />
              {input && (
                <button
                  type="button"
                  onClick={() => setInput("")}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl hover:from-blue-600 hover:to-cyan-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
              <span className="hidden sm:inline">Send</span>
            </button>
          </div>
          <div className="mt-3 text-xs text-gray-600 dark:text-gray-300 text-center flex items-center justify-center space-x-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>MedRAG is for educational purposes only. Always consult a healthcare professional for medical advice.</span>
          </div>
        </form>
      </div>
    </div>
  );
}

