import React from 'react';
import { NavLink } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Activity,
  Database,
  Code,
  Brain,
  Share2,
  MessageSquare,
  Menu,
  X,
  Sparkles,
  Folder,
  Search,
  GitBranch,
  BarChart3,
  Zap,
  Server
} from 'lucide-react';

interface NavigationProps {
  isCollapsed: boolean;
  onToggle: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({ isCollapsed, onToggle }) => {
  const navigationItems = [
    {
      to: '/gemini-server',
      icon: Server,
      label: 'Gemini Server',
      description: 'Integrated AI with context management',
      badge: 'Enhanced'
    },
    {
      to: '/dashboard',
      icon: Activity,
      label: 'Agent Dashboard',
      description: 'Monitor and manage AI agents',
      badge: null
    },
    {
      to: '/auto-claude',
      icon: Zap,
      label: 'Auto-Claude',
      description: 'TypeScript-aware code fixing',
      badge: 'New'
    },
    {
      to: '/monitoring',
      icon: BarChart3,
      label: 'System Monitoring',
      description: 'Performance & health metrics',
      badge: 'Live'
    },
    {
      to: '/context',
      icon: Folder,
      label: 'Context Analysis',
      description: 'Analyze project structure',
      badge: null
    },
    {
      to: '/generation',
      icon: Code,
      label: 'Code Generation',
      description: 'AI-powered code creation',
      badge: 'Pro'
    },
    {
      to: '/memory',
      icon: Brain,
      label: 'Memory RAG',
      description: 'Semantic search & memory',
      badge: null
    },
    {
      to: '/knowledge-graph',
      icon: Share2,
      label: 'Knowledge Graph',
      description: 'Visualize relationships',
      badge: null
    },
    {
      to: '/chat',
      icon: MessageSquare,
      label: 'Legacy Chat',
      description: 'Original chat interface',
      badge: null
    }
  ];

  return (
    <div className={`fixed left-0 top-0 h-full bg-neutral-800 border-r border-neutral-700 transition-all duration-300 z-50 ${
      isCollapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-neutral-700">
        {!isCollapsed && (
          <div className="flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-blue-400" />
            <h1 className="text-lg font-bold text-neutral-100">Gemini Agents</h1>
          </div>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={onToggle}
          className="text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700"
        >
          {isCollapsed ? <Menu className="w-4 h-4" /> : <X className="w-4 h-4" />}
        </Button>
      </div>

      {/* Navigation Items */}
      <nav className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `group flex items-center gap-3 px-3 py-2 rounded-md transition-colors relative ${
                    isActive
                      ? 'bg-blue-500/20 text-blue-300 border-r-2 border-blue-500'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50'
                  } ${isCollapsed ? 'justify-center' : ''}`
                }
              >
                <Icon className={`flex-shrink-0 ${isCollapsed ? 'w-5 h-5' : 'w-5 h-5'}`} />

                {!isCollapsed && (
                  <>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">{item.label}</span>
                        {item.badge && (
                          <Badge className="text-xs bg-blue-500/20 text-blue-300 border-blue-500/30">
                            {item.badge}
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-neutral-500 truncate">
                        {item.description}
                      </p>
                    </div>
                  </>
                )}

                {/* Tooltip for collapsed state */}
                {isCollapsed && (
                  <div className="absolute left-full ml-2 px-2 py-1 bg-neutral-900 text-neutral-200 text-sm rounded-md border border-neutral-700 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50">
                    <div className="font-medium">{item.label}</div>
                    <div className="text-xs text-neutral-400">{item.description}</div>
                    <div className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1 border-4 border-transparent border-r-neutral-900"></div>
                  </div>
                )}
              </NavLink>
            );
          })}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-neutral-700">
        {!isCollapsed ? (
          <div className="text-center">
            <p className="text-xs text-neutral-500">
              Gemini Agent System v1.0
            </p>
            <p className="text-xs text-neutral-600 mt-1">
              Powered by Claude Code
            </p>
          </div>
        ) : (
          <div className="flex justify-center">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          </div>
        )}
      </div>
    </div>
  );
};
