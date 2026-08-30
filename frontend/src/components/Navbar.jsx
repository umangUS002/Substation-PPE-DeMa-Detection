import React, { useState, useEffect } from 'react';
import { ShieldCheck, Volume2, VolumeX, Zap } from 'lucide-react';

export default function Navbar({ analytics, soundEnabled, setSoundEnabled }) {
  const [time, setTime] = useState('');

  useEffect(() => {
    const tick = () => setTime(new Date().toLocaleTimeString('en-IN', { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const compliance    = analytics?.compliance_rate ?? 97.4;
  const criticalAlerts = analytics?.unresolved_alerts || 0;
  const totalViolations = analytics?.total_violations || 0;

  return (
    <header className="navbar">
      <div className="brand-section">
        <div className="brand-logo">
          <ShieldCheck size={18} style={{ color: '#22d3ee' }} />
        </div>
        <div className="brand-title">
          PPE Safety Monitor
          <span className="brand-badge">
            <Zap size={9} style={{ display: 'inline', marginRight: 3 }} />
            LIVE
          </span>
        </div>
      </div>

      <div className="header-metrics">
        {/* Real-time clock */}
        <div className="metric-pill" title="System Time">
          <span className="metric-label">System Time</span>
          <span className="metric-value" style={{ fontFamily: 'var(--mono)', fontSize: 14 }}>
            {time || '—:—:—'}
          </span>
        </div>

        {/* Compliance */}
        <div className="metric-pill" title="Plant Compliance Rate">
          <span className="metric-label">Compliance</span>
          <span className={`metric-value ${compliance >= 90 ? 'safe' : 'warning'}`}>
            {compliance}%
          </span>
        </div>

        {/* Unresolved alerts */}
        {criticalAlerts > 0 && (
          <div className="metric-pill" style={{ borderColor: 'rgba(244,63,94,0.3)' }}>
            <span className="metric-label">Unresolved</span>
            <span className="metric-value warning">{criticalAlerts}</span>
          </div>
        )}

        {/* Sound toggle */}
        <button
          className="btn-icon"
          title={soundEnabled ? 'Mute alarm' : 'Enable alarm'}
          onClick={() => setSoundEnabled(!soundEnabled)}
          style={{
            borderColor: soundEnabled ? 'rgba(245,158,11,0.4)' : undefined,
            color:        soundEnabled ? 'var(--amber)'        : undefined,
          }}
        >
          {soundEnabled ? <Volume2 size={15} /> : <VolumeX size={15} />}
        </button>
      </div>
    </header>
  );
}
