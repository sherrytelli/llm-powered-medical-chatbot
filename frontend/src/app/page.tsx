"use client";

import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <div className="max-w-2xl mx-auto mt-10">
      <h1 className="text-3xl font-bold text-center mb-6">Medical RAG Assistant</h1>
      <ChatInterface />
    </div>
  );
}

