"use client";

import { useState, useEffect } from "react";
import axios from "axios";

interface Config {
  kb_path: string;
  index_path: string;
  chunks_path: string;
  embed_model: string;
  chat_model: string;
  k: number;
}

interface StatusMessage {
  type: "success" | "error" | "info";
  text: string;
}

export default function AdminPanel() {
  const [config, setConfig] = useState<Config>({
    kb_path: "knowledge/medquad.csv",
    index_path: "knowledge/faiss.index",
    chunks_path: "knowledge/chunks.json",
    embed_model: "nomic-embed-text:v1.5",
    chat_model: "phi3.5:3.8b",
    k: 3,
  });
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isRebuilding, setIsRebuilding] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await axios.get("http://localhost:8000/admin/config");
      setConfig(res.data);
    } catch (error) {
      console.error("Failed to fetch config", error);
      showStatus("error", "Failed to load configuration");
    }
  };

  const showStatus = (type: "success" | "error" | "info", text: string) => {
    setStatus({ type, text });
    setTimeout(() => setStatus(null), 5000);
  };

  const handleConfigChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name === 'k' ? parseInt(value) || 0 : value
    }));
  };

  const handleUpdateConfig = async () => {
    try {
      const formData = new FormData();
      formData.append("kb_path", config.kb_path);
      formData.append("index_path", config.index_path);
      formData.append("chunks_path", config.chunks_path);
      formData.append("embed_model", config.embed_model);
      formData.append("chat_model", config.chat_model);
      formData.append("k", config.k.toString());

      await axios.post("http://localhost:8000/admin/update-config", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      showStatus("success", "Configuration updated successfully!");
    } catch (error) {
      console.error("Config update error:", error);
      showStatus("error", "Failed to update configuration");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleRebuildIndex = async () => {
    if (!file) {
      showStatus("error", "Please select a CSV file first");
      return;
    }

    setIsRebuilding(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("kb_path", config.kb_path);
      formData.append("index_path", config.index_path);
      formData.append("chunks_path", config.chunks_path);
      formData.append("embed_model", config.embed_model);
      formData.append("chat_model", config.chat_model);
      formData.append("k", config.k.toString());

      await axios.post("http://localhost:8000/admin/rebuild-index", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      showStatus("success", "Index rebuilt successfully! The system is now using the new dataset.");
      setFile(null);
    } catch (error) {
      console.error("Rebuild error:", error);
      showStatus("error", "Failed to rebuild index");
    } finally {
      setIsRebuilding(false);
    }
  };

  const getStatusIcon = (type: string) => {
    switch (type) {
      case "success":
        return (
          <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case "error":
        return (
          <svg className="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="text-center mb-8 animate-slide-up">
        <div className="inline-flex items-center space-x-2 px-4 py-2 bg-blue-100 dark:bg-blue-900/30 rounded-full text-sm text-blue-700 dark:text-blue-300 mb-4">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          </svg>
          <span>System Configuration</span>
        </div>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
          Admin Panel
        </h1>
        <p className="text-gray-600 dark:text-gray-300">Configure and manage your Medical RAG system</p>
      </div>

      {/* Status Messages */}
      {status && (
        <div className={`mb-6 p-4 rounded-xl flex items-center space-x-3 animate-slide-up ${
          status.type === "success" ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800" :
          status.type === "error" ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800" :
          "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
        }`}>
          {getStatusIcon(status.type)}
          <span className={`text-sm font-medium ${
            status.type === "success" ? "text-green-800 dark:text-green-200" :
            status.type === "error" ? "text-red-800 dark:text-red-200" :
            "text-blue-800 dark:text-blue-200"
          }`}>
            {status.text}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Configuration Panel */}
        <div className="glass rounded-2xl shadow-lg overflow-hidden animate-slide-up" style={{ animationDelay: '100ms' }}>
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-gray-800 dark:to-gray-700">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-500 p-3 rounded-xl shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Configuration</h2>
                <p className="text-sm text-gray-600 dark:text-gray-300">Update model and system settings</p>
              </div>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Chat Model
                </label>
                <select
                  name="chat_model"
                  value={config.chat_model}
                  onChange={handleConfigChange}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                >
                  <option value="phi3.5:3.8b">Phi 3.5 (3.8B)</option>
                  <option value="llama3.2">Llama 3.2</option>
                  <option value="gemma2:2b">Gemma 2 (2B)</option>
                  <option value="mistral">Mistral</option>
                  <option value="llama2">Llama 2</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Embedding Model
                </label>
                <select
                  name="embed_model"
                  value={config.embed_model}
                  onChange={handleConfigChange}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                >
                  <option value="nomic-embed-text:v1.5">Nomic Embed Text v1.5</option>
                  <option value="all-minilm">All-MiniLM</option>
                  <option value="bge-m3">BGE M3</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Top-K Results
                </label>
                <input
                  type="number"
                  name="k"
                  value={config.k}
                  onChange={handleConfigChange}
                  min="1"
                  max="10"
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  KB Path
                </label>
                <input
                  type="text"
                  name="kb_path"
                  value={config.kb_path}
                  onChange={handleConfigChange}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Index Path
                </label>
                <input
                  type="text"
                  name="index_path"
                  value={config.index_path}
                  onChange={handleConfigChange}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Chunks Path
                </label>
                <input
                  type="text"
                  name="chunks_path"
                  value={config.chunks_path}
                  onChange={handleConfigChange}
                  className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-all duration-200 hover:border-blue-400"
                />
              </div>
            </div>

            <button
              onClick={handleUpdateConfig}
              className="w-full px-6 py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-xl hover:from-blue-600 hover:to-cyan-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <span className="font-semibold">Update Configuration</span>
            </button>
          </div>
        </div>

        {/* Index Rebuild Panel */}
        <div className="glass rounded-2xl shadow-lg overflow-hidden animate-slide-up" style={{ animationDelay: '200ms' }}>
          <div className="p-6 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-gray-800 dark:to-gray-700">
            <div className="flex items-center space-x-3">
              <div className="bg-green-500 p-3 rounded-xl shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Rebuild Index</h2>
                <p className="text-sm text-gray-600 dark:text-gray-300">Upload new dataset to rebuild the knowledge base</p>
              </div>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl p-4">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-yellow-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3 flex-1">
                  <h3 className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">Important Warning</h3>
                  <div className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
                    <p>Rebuilding the index will replace the current knowledge base. This process may take several minutes depending on the dataset size.</p>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Upload CSV Dataset
              </label>
              <div
                className={`mt-1 flex flex-col items-center justify-center px-6 py-8 border-2 border-dashed rounded-xl transition-all duration-200 ${
                  isDragging 
                    ? "border-blue-400 bg-blue-50 dark:border-blue-500 dark:bg-blue-900/20" 
                    : "border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="space-y-3 text-center">
                  <div className="mx-auto w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <div>
                    <label htmlFor="file-upload" className="relative cursor-pointer bg-transparent rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none">
                      <span className="text-lg">Upload a CSV file</span>
                      <input
                        id="file-upload"
                        name="file-upload"
                        type="file"
                        className="sr-only"
                        accept=".csv"
                        onChange={handleFileUpload}
                      />
                    </label>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">or drag and drop here</p>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    CSV files only • Maximum size: 10MB
                  </p>
                </div>
              </div>
              
              {file && (
                <div className="mt-4 flex items-center justify-between p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-200 dark:border-blue-800">
                  <div className="flex items-center space-x-3">
                    <div className="bg-blue-500 p-2 rounded-lg">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-blue-800 dark:text-blue-200">{file.name}</p>
                      <p className="text-xs text-gray-600 dark:text-gray-300">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setFile(null)}
                    className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 p-2 hover:bg-blue-100 dark:hover:bg-blue-800 rounded-lg transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              )}
            </div>

            <button
              onClick={handleRebuildIndex}
              disabled={!file || isRebuilding}
              className="w-full px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-xl hover:from-green-600 hover:to-emerald-600 focus:ring-2 focus:ring-green-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-2 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
            >
              {isRebuilding ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span className="font-semibold">Rebuilding Index...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span className="font-semibold">Upload & Rebuild Index</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

