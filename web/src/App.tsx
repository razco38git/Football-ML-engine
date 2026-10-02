import { useState } from 'react';
import MatchPredictor from './components/MatchPredictor';
import PlayerDatabase from './components/PlayerDatabase';
import PlayerSimilarity from './components/PlayerSimilarity';
import TeamStrength from './components/TeamStrength';
import PredictionHistory from './components/PredictionHistory';
import SeasonProjection from './components/SeasonProjection';
import WhatIf from './components/WhatIf';

type Tab = 'predictor' | 'whatif' | 'players' | 'similarity' | 'teams' | 'projection' | 'history';

const tabs: { id: Tab; label: string; icon: string; badge?: string }[] = [
  { id: 'predictor', label: 'Match Predictor', icon: '⚡' },
  { id: 'players', label: 'Player Ratings', icon: '👤', badge: 'EA FC' },
  { id: 'similarity', label: 'Player Similarity', icon: '🔍' },
  { id: 'teams', label: 'Team Strength', icon: '🏆' },
  { id: 'projection', label: 'Projected Tables', icon: '🔮' },
  { id: 'history', label: 'Prediction Accuracy', icon: '📊' },
  // Last, and badged, because it is the one tab whose answers can never be
  // checked: these pairings are not scheduled, so no result will ever settle
  // them. Everything above describes matches that did or will happen.
  { id: 'whatif', label: 'What If?', icon: '🧪', badge: 'SANDBOX' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('predictor');

  return (
    <div className="min-h-screen" style={{ background: 'var(--background)' }}>
      {/* Top header */}
      <header
        className="sticky top-0 z-40 px-6 py-0"
        style={{
          background: 'rgba(7,9,15,0.92)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--border)',
        }}
      >
        <div className="max-w-7xl mx-auto flex items-center">
          {/* Logo */}
          <div className="flex items-center gap-3 py-3 mr-8 flex-shrink-0">
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center text-lg font-display font-black"
              style={{ background: 'linear-gradient(135deg, #00e676, #00b0ff)', color: '#000' }}
            >
              ⚽
            </div>
            <div>
              <div className="font-display font-black text-lg leading-tight" style={{ color: '#00e676' }}>
                FootballIQ
              </div>
              <div className="text-xs leading-tight" style={{ color: 'var(--muted-foreground)' }}>
                ML Analytics Platform
              </div>
            </div>
          </div>

          {/* Nav tabs */}
          <nav className="flex items-stretch gap-0.5 flex-1 overflow-x-auto">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className="flex items-center gap-2 px-4 py-4 text-sm font-display font-bold transition-all relative flex-shrink-0"
                style={{
                  color: activeTab === tab.id ? '#00e676' : 'var(--muted-foreground)',
                  borderBottom: activeTab === tab.id ? '2px solid #00e676' : '2px solid transparent',
                  background: 'transparent',
                }}
              >
                <span className="text-base">{tab.icon}</span>
                <span>{tab.label}</span>
                {tab.badge && (
                  <span
                    className="text-xs px-1.5 py-0.5 rounded font-bold ml-0.5"
                    style={{ background: '#00e67622', color: '#00e676', fontSize: 9 }}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>

          {/* Live badge */}
          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-full ml-4 flex-shrink-0"
            style={{ background: 'rgba(0,230,118,0.1)', border: '1px solid rgba(0,230,118,0.25)' }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#00e676', boxShadow: '0 0 6px #00e676', animation: 'pulse 2s infinite' }} />
            <span className="text-xs font-display font-bold" style={{ color: '#00e676' }}>LIVE</span>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'predictor' && <MatchPredictor />}
        {activeTab === 'whatif' && <WhatIf />}
        {activeTab === 'players' && <PlayerDatabase />}
        {activeTab === 'similarity' && <PlayerSimilarity />}
        {activeTab === 'teams' && <TeamStrength />}
        {activeTab === 'projection' && <SeasonProjection />}
        {activeTab === 'history' && <PredictionHistory />}
      </main>

      {/* Footer */}
      <footer
        className="mt-12 px-6 py-4 text-center"
        style={{ borderTop: '1px solid var(--border)' }}
      >
        <p className="text-xs" style={{ color: 'var(--muted-foreground)' }}>
          FootballIQ · football-data.co.uk, Understat, FBref and EA FC · For analytical purposes only
        </p>
      </footer>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
