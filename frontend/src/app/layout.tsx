import type { Metadata } from "next";
import "./globals.css";
import Link from "next/link";
import { ThemeToggle } from "@/components/ThemeToggle";

export const metadata: Metadata = {
  title: "MedRAG - Medical RAG Assistant",
  description: "AI-powered medical information assistant with RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 text-gray-900 dark:text-white transition-colors duration-300">
        {/* Animated background pattern */}
        <div className="fixed inset-0 z-0 opacity-30 dark:opacity-10 pointer-events-none">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(59,130,246,0.1),transparent_50%)]" />
          <div className="absolute top-0 left-0 w-96 h-96 bg-blue-200 dark:bg-blue-900 rounded-full mix-blend-multiply dark:mix-blend-screen filter blur-3xl opacity-20 animate-float" />
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-200 dark:bg-cyan-900 rounded-full mix-blend-multiply dark:mix-blend-screen filter blur-3xl opacity-20 animate-float" style={{ animationDelay: '1s' }} />
        </div>

        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 glass">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16 items-center">
              <Link href="/" className="flex items-center space-x-3 group">
                <div className="relative">
                  <div className="absolute inset-0 bg-blue-500 rounded-lg blur opacity-50 group-hover:opacity-75 transition-opacity" />
                  <div className="relative bg-gradient-to-br from-blue-600 to-cyan-600 p-2 rounded-lg shadow-lg">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                </div>
                <div>
                  <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                    MedRAG
                  </span>
                  <span className="hidden sm:block text-sm text-gray-600 dark:text-gray-300">Medical AI Assistant</span>
                </div>
              </Link>
              
              <div className="flex items-center space-x-2">
                <Link 
                  href="/" 
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-gray-800 transition-all duration-200"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <span>Chat</span>
                </Link>
                <Link 
                  href="/admin" 
                  className="flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-gray-800 transition-all duration-200"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span>Admin</span>
                </Link>
                <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-2" />
                <ThemeToggle />
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="relative z-10 pt-16 min-h-screen">
          {children}
        </main>

        {/* Footer */}
        <footer className="relative z-10 glass border-t border-gray-200 dark:border-gray-700 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
              <div className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>For educational purposes only. Not a substitute for professional medical advice.</span>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Powered by Ollama & RAG
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}

