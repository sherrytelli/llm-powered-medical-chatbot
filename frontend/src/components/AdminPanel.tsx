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

export default function AdminPanel() {
  const [config, setConfig] = useState<Config>({
    kb_path: "knowledge/medquad.csv",
    index_path: "knowledge/faiss.index",
    chunks_path: "knowledge/chunks.json",
    embed_model: "nomic-embed-text:v1.5",
    chat_model: "phi3.5:3.8b",
    k: 3,
  });
  const [status, setStatus] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await axios.get("http://localhost:8000/admin/config");
      setConfig(res.data);
    } catch (error) {
      console.error("Failed to fetch config", error);
    }
  };

  const handleConfigChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setConfig({ ...config, [e.target.name]: e.target.value });
  };

  const handleUpdateConfig = async () => {
    setStatus("Updating configuration...");
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
      setStatus("Configuration updated successfully!");
      setTimeout(() => setStatus(""), 3000);
    } catch (error) {
      setStatus("Error updating configuration.");
      setTimeout(() => setStatus(""), 3000);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    setFile(e.target.files[0]);
  };

  const handleRebuildIndex = async () => {
    if (!file) {
      setStatus("Please select a CSV file first.");
      return;
    }

    setStatus("Uploading file and rebuilding index...");
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
      setStatus("Index rebuilt successfully!");
      setTimeout(() => setStatus(""), 5000);
    } catch (error) {
      setStatus("Error rebuilding index.");
      setTimeout(() => setStatus(""), 3000);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Chat Model</label>
          <select
            name="chat_model"
            value={config.chat_model}
            onChange={handleConfigChange}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
          >
            <option value="phi3.5:3.8b">phi3.5:3.8b</option>
            <option value="llama3.2">llama3.2</option>
            <option value="gemma2:2b">gemma2:2b</option>
            <option value="mistral">mistral</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Embed Model</label>
          <select
            name="embed_model"
            value={config.embed_model}
            onChange={handleConfigChange}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
          >
            <option value="nomic-embed-text:v1.5">nomic-embed-text:v1.5</option>
            <option value="all-minilm">all-minilm</option>
            <option value="bge-m3">bge-m3</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">K (Top-K Results)</label>
          <input
            type="number"
            name="k"
            value={config.k}
            onChange={handleConfigChange}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">KB Path</label>
          <input
            type="text"
            name="kb_path"
            value={config.kb_path}
            onChange={handleConfigChange}
            className="w-full p-2 border rounded dark:bg-gray-700 dark:border-gray-600"
          />
        </div>
      </div>

      <button
        onClick={handleUpdateConfig}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Update Configuration
      </button>

      <hr className="my-4" />

      <div>
        <h3 className="text-xl font-bold mb-2">Rebuild Index from CSV</h3>
        <input
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mb-4"
        />
        {file && <p className="text-sm text-gray-500 mb-2">Selected: {file.name}</p>}
        <button
          onClick={handleRebuildIndex}
          disabled={!file}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
        >
          Upload & Rebuild Index
        </button>
      </div>

      {status && (
        <div className="mt-4 p-3 bg-yellow-100 text-yellow-800 rounded">
          {status}
        </div>
      )}
    </div>
  );
}

