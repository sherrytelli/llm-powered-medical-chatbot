"use client";

import AdminPanel from "@/components/AdminPanel";

export default function AdminPage() {
  return (
    <div className="max-w-4xl mx-auto mt-10">
      <h1 className="text-3xl font-bold text-center mb-6">Admin Configuration</h1>
      <AdminPanel />
    </div>
  );
}

