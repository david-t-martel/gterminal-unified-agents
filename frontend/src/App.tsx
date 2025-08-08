import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navigation } from '@/components/Navigation';
import { AgentDashboard } from '@/components/AgentDashboard';
import { AutoClaudeDashboard } from '@/components/AutoClaudeDashboard';
import { MonitoringDashboard } from '@/components/MonitoringDashboard';
import { ContextAnalysis } from '@/components/ContextAnalysis';
import { CodeGeneration } from '@/components/CodeGeneration';
import { MemoryRAG } from '@/components/MemoryRAG';
import { KnowledgeGraph } from '@/components/KnowledgeGraph';
import { LegacyChat } from '@/components/LegacyChat';
import { GeminiServerDashboard } from '@/components/GeminiServerDashboard';

export default function App() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <Router>
      <div className="flex h-screen bg-neutral-900 text-neutral-100 font-sans antialiased">
        <Navigation isCollapsed={isCollapsed} onToggle={() => setIsCollapsed(!isCollapsed)} />

        <main className={`flex-1 overflow-hidden transition-all duration-300 ${isCollapsed ? 'ml-16' : 'ml-64'}`}>
          <Routes>
            <Route path="/" element={<Navigate to="/gemini-server" replace />} />
            <Route path="/dashboard" element={<AgentDashboard />} />
            <Route path="/gemini-server" element={<GeminiServerDashboard />} />
            <Route path="/auto-claude" element={<AutoClaudeDashboard />} />
            <Route path="/monitoring" element={<MonitoringDashboard />} />
            <Route path="/context" element={<ContextAnalysis />} />
            <Route path="/generation" element={<CodeGeneration />} />
            <Route path="/memory" element={<MemoryRAG />} />
            <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
            <Route path="/chat" element={<LegacyChat />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
