import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Medical RAG Chatbot",
  description: "Educational Medical RAG Assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <nav className="bg-gray-800 p-4">
          <div className="container mx-auto flex justify-between items-center">
            <Link href="/" className="text-white text-xl font-bold">
              🏥 MedRAG
            </Link>
            <div>
              <Link href="/" className="text-gray-300 hover:text-white mr-4">
                Chat
              </Link>
              <Link href="/admin" className="text-gray-300 hover:text-white">
                Admin
              </Link>
            </div>
          </div>
        </nav>
        <main className="container mx-auto p-4">
          {children}
        </main>
      </body>
    </html>
  );
}

