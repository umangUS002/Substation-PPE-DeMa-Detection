import React, { useState, useEffect } from 'react';
import { RefreshCw, CheckCircle2, X, AlertTriangle, Clock } from 'lucide-react';

const API = 'http://localhost:8000';

function SeverityBadge({ level }) {
  const colors = {
    CRITICAL: { bg: 'rgba(244,63,94,0.15)',   color: '#f43f5e', border: 'rgba(244,63,94,0.35)' },
    HIGH:     { bg: 'rgba(245,158,11,0.15)',  color: '#f59e0b', border: 'rgba(245,158,11,0.3)' },
    MEDIUM:   { bg: 'rgba(34,211,238,0.12)',  color: '#22d3ee', border: 'rgba(34,211,238,0.25)' },
  };
  const c = colors[level] || colors.MEDIUM;
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 99,
      fontSize: 10, fontWeight: 700, background: c.bg, color: c.color,
      border: `1px solid ${c.border}`, letterSpacing: '0.2px',
    }}>{level}</span>
  );
}

export default function IncidentStream({ onIncidentResolved }) {
  const [incidents, setIncidents] = useState([]);
  const [loading,   setLoading]   = useState(false);
  const [lightbox,  setLightbox]  = useState(null);
  const [camFilter, setCamFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');

  const fetchIncidents = async () => {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/violations?limit=80&_t=${Date.now()}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setIncidents(data.violations || []);
    } catch (e) {
      console.error('IncidentStream fetch failed:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIncidents();
    const id = setInterval(fetchIncidents, 4000);
    return () => clearInterval(id);
  }, []);

  const handleResolve = async (eventId) => {
    try {
      await fetch(`${API}/api/violations/${eventId}/resolve`, { method: 'POST' });
      fetchIncidents();
      if (onIncidentResolved) onIncidentResolved();
    } catch (e) {
      console.error('Resolve error:', e);
    }
  };

  const uniqueCams = Array.from(new Set(incidents.map(i => i.camera_id))).sort();

  const filtered = incidents.filter(v => {
    if (camFilter !== 'ALL' && v.camera_id !== camFilter) return false;
    if (statusFilter !== 'ALL' && v.status !== statusFilter) return false;
    return true;
  });

  const unresolvedCount = incidents.filter(v => v.status === 'UNRESOLVED').length;

  return (
    <div className="fade-up">
      {/* ── TOOLBAR ── */}
      <div className="toolbar-bar">
        <div className="filter-group">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Clock size={15} style={{ color: 'var(--cyan)' }} />
            <span style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-1)' }}>
              Incident Log
            </span>
          </div>
          <span className="count-tag">{filtered.length} events</span>
          {unresolvedCount > 0 && (
            <span style={{
              background: 'var(--red-dim)', color: 'var(--red)',
              border: '1px solid rgba(244,63,94,0.3)',
              padding: '2px 10px', borderRadius: 99, fontSize: 11, fontWeight: 700,
            }}>
              {unresolvedCount} unresolved
            </span>
          )}
        </div>

        <div className="filter-group">
          <select className="filter-select" value={camFilter} onChange={e => setCamFilter(e.target.value)}>
            <option value="ALL">All Cameras</option>
            {uniqueCams.map(c => <option key={c} value={c}>{c}</option>)}
          </select>

          <select className="filter-select" value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="ALL">All Status</option>
            <option value="UNRESOLVED">Unresolved</option>
            <option value="RESOLVED">Resolved</option>
          </select>

          <button className="btn-primary" onClick={fetchIncidents} disabled={loading}>
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* ── TABLE ── */}
      <div className="data-table-container">
        {/* Loading bar */}
        {loading && (
          <div style={{
            height: 2,
            background: 'linear-gradient(90deg, var(--cyan), var(--violet), var(--cyan))',
            backgroundSize: '200% 100%',
            animation: 'shimmer 1.4s linear infinite',
          }} />
        )}

        <table className="data-table">
          <thead>
            <tr>
              <th style={{ width: 88 }}>Snapshot</th>
              <th>Time</th>
              <th>Camera</th>
              <th>Missing PPE</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '52px 0', color: 'var(--text-3)' }}>
                  <CheckCircle2 size={28} style={{ color: 'var(--emerald)', marginBottom: 8, display: 'block', margin: '0 auto 10px' }} />
                  No incidents match current filters.
                </td>
              </tr>
            ) : (
              filtered.map(v => (
                <tr key={v.event_id}>
                  {/* Snapshot */}
                  <td>
                    {v.snapshot_url ? (
                      <div
                        className="table-snapshot-thumb"
                        onClick={() => setLightbox(`${API}${v.snapshot_url}`)}
                        title="Click to enlarge"
                      >
                        <img src={`${API}${v.snapshot_url}`} alt="Evidence" className="thumb-img" />
                      </div>
                    ) : (
                      <div style={{
                        width: 68, height: 42, background: 'var(--bg-elevated)',
                        borderRadius: 5, border: '1px solid var(--border)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                      }}>
                        <AlertTriangle size={14} style={{ color: 'var(--text-3)' }} />
                      </div>
                    )}
                  </td>

                  {/* Time */}
                  <td style={{ color: 'var(--text-2)', fontSize: 12, whiteSpace: 'nowrap' }}>
                    {v.datetime_str}
                  </td>

                  {/* Camera */}
                  <td>
                    <div style={{ fontWeight: 600, fontSize: 12.5 }}>{v.camera_id}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{v.zone}</div>
                  </td>

                  {/* Missing PPE */}
                  <td>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {(v.missing_ppe || []).map((item, i) => (
                        <span key={i} style={{
                          background: 'var(--red-dim)', color: 'var(--red)',
                          border: '1px solid rgba(244,63,94,0.3)',
                          fontSize: 10, fontWeight: 600, padding: '1px 7px', borderRadius: 99,
                        }}>
                          {item}
                        </span>
                      ))}
                      {(!v.missing_ppe || v.missing_ppe.length === 0) && (
                        <span style={{ color: 'var(--text-3)', fontSize: 12 }}>—</span>
                      )}
                    </div>
                  </td>

                  {/* Severity */}
                  <td><SeverityBadge level={v.severity || 'HIGH'} /></td>

                  {/* Status */}
                  <td>
                    <span className={`status-badge ${v.status === 'UNRESOLVED' ? 'violation' : 'compliant'}`}>
                      {v.status === 'UNRESOLVED' ? 'Open' : 'Resolved'}
                    </span>
                  </td>

                  {/* Action */}
                  <td>
                    {v.status === 'UNRESOLVED' ? (
                      <button className="action-btn-small" onClick={() => handleResolve(v.event_id)}>
                        Resolve
                      </button>
                    ) : (
                      <span style={{ color: 'var(--emerald)', fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
                        <CheckCircle2 size={12} /> Done
                      </span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* ── LIGHTBOX ── */}
      {lightbox && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            background: 'rgba(0,0,0,0.92)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          onClick={() => setLightbox(null)}
        >
          <button
            onClick={() => setLightbox(null)}
            style={{
              position: 'absolute', top: 20, right: 24,
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '50%', padding: 8, cursor: 'pointer', color: '#fff',
            }}
          >
            <X size={18} />
          </button>
          <img
            src={lightbox}
            alt="Evidence"
            style={{ maxWidth: '88vw', maxHeight: '85vh', borderRadius: 10, boxShadow: '0 24px 80px rgba(0,0,0,0.9)' }}
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
