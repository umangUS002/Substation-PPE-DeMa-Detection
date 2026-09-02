import React, { useState, useEffect, useRef, useCallback } from 'react';
import { X, Users, RefreshCw, Clock, AlertTriangle, CheckCircle2, ZoomIn, Download } from 'lucide-react';

// Call backend directly to bypass Vite proxy ECONNABORTED errors from MJPEG streams
const API = 'http://localhost:8000';

export default function SpotlightModal({ camera, onClose }) {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [selectedSnap, setSelectedSnap] = useState(null);
  const [streamKey, setStreamKey]   = useState(Date.now());
  const intervalRef = useRef(null);

  const fetchViolations = useCallback(async () => {
    if (!camera?.camera_id) return;
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(
        `${API}/api/cameras/${camera.camera_id}/violations?limit=30&_t=${Date.now()}`
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setViolations(data.violations || []);
    } catch (e) {
      setError(`${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [camera?.camera_id]);

  useEffect(() => {
    if (!camera?.camera_id) return;
    setViolations([]);
    setError(null);
    fetchViolations();
    intervalRef.current = setInterval(fetchViolations, 4000);
    return () => clearInterval(intervalRef.current);
  }, [camera?.camera_id, fetchViolations]);

  if (!camera) return null;

  const isViolation = camera.status === 'VIOLATION';
  const workerCount = camera.total_detected_workers || 0;
  const missing     = camera.missing_ppe || [];

  const sevColor = (s) =>
    s === 'CRITICAL' ? '#ef4444' : s === 'HIGH' ? '#f59e0b' : '#06b6d4';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-box spotlight-modal-clean"
        onClick={e => e.stopPropagation()}
        style={{
          maxWidth: 1080, width: '95vw', maxHeight: '92vh',
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}
      >
        {/* HEADER */}
        <div className="modal-header" style={{ flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span className="spotlight-cam-badge">{camera.camera_id}</span>
            <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--text-primary)' }}>
              {camera.name}
            </span>
            {camera.is_live && <span className="live-pill-pulse">🔴 LIVE</span>}
            <span className={`status-badge ${isViolation ? 'violation' : 'compliant'}`} style={{ fontSize: 11 }}>
              {isViolation ? '⚠ VIOLATION' : '✅ COMPLIANT'}
            </span>
            <span className="worker-count-pill">
              <Users size={12} style={{ display: 'inline', marginRight: 4 }} />
              {workerCount} Worker{workerCount !== 1 ? 's' : ''}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn-icon" title="Refresh stream" onClick={() => setStreamKey(Date.now())}>
              <RefreshCw size={14} />
            </button>
            <button className="btn-icon" title="Close" onClick={onClose}>
              <X size={16} />
            </button>
          </div>
        </div>

        {/* BODY ROW */}
        <div style={{ display: 'flex', gap: 16, padding: '12px 16px', flexShrink: 0, flexWrap: 'wrap' }}>

          {/* LIVE FEED */}
          <div style={{ flex: '1 1 340px', minWidth: 260 }}>
            <div style={{
              borderRadius: 10, overflow: 'hidden', position: 'relative', background: '#0a0e17',
              border: `2px solid ${isViolation ? 'rgba(239,68,68,0.6)' : 'rgba(16,185,129,0.4)'}`,
              boxShadow: isViolation ? '0 0 24px rgba(239,68,68,0.18)' : '0 0 14px rgba(16,185,129,0.1)',
            }}>
              <img
                key={streamKey}
                src={`${API}/api/camera/${camera.camera_id}/feed`}
                alt={camera.name}
                style={{ width: '100%', display: 'block', maxHeight: 250, objectFit: 'cover' }}
              />
              {isViolation && (
                <div style={{
                  position: 'absolute', top: 8, right: 8,
                  background: 'rgba(239,68,68,0.88)', borderRadius: 6,
                  padding: '2px 8px', fontSize: 10, fontWeight: 700, color: '#fff',
                }}>⚠ ALERT</div>
              )}
            </div>

            {isViolation && missing.length > 0 && (
              <div style={{
                marginTop: 8, padding: '8px 12px', borderRadius: 8,
                background: 'rgba(239,68,68,0.09)', border: '1px solid rgba(239,68,68,0.35)',
              }}>
                <div style={{ fontSize: 10, color: '#f87171', fontWeight: 700, marginBottom: 5, letterSpacing: '0.05em' }}>
                  MISSING PPE
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {missing.map((item, i) => (
                    <span key={i} style={{
                      background: 'rgba(239,68,68,0.18)', border: '1px solid rgba(239,68,68,0.45)',
                      color: '#fca5a5', padding: '2px 9px', borderRadius: 99, fontSize: 11, fontWeight: 600,
                    }}>NO {item}</span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* ZONE INFO CARDS */}
          <div style={{ flex: '1 1 190px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              ['Zone', camera.zone],
              ['Hazard Level', (camera.hazard_level || '').replace(/_/g, ' ')],
              ['Mandatory PPE', (camera.required_ppe || ['Helmet', 'Footwear']).join(', ')],
              ['Workers Detected', workerCount],
              ['Alert Consensus', camera.temporal_consensus || '—'],
            ].map(([label, val]) => (
              <div key={label} style={{
                padding: '8px 12px', borderRadius: 8,
                background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)',
              }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {label}
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>{val}</div>
              </div>
            ))}
          </div>
        </div>

        {/* LOG SECTION */}
        <div style={{
          flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column',
          borderTop: '1px solid rgba(255,255,255,0.08)', padding: '0 16px',
        }}>
          {/* Log header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 0', flexShrink: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Clock size={14} style={{ color: 'var(--accent-cyan)' }} />
              <span style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>
                Violation Snapshot History
              </span>
              <span style={{
                background: violations.length > 0 ? 'rgba(239,68,68,0.15)' : 'rgba(255,255,255,0.06)',
                color: violations.length > 0 ? '#f87171' : 'var(--text-muted)',
                border: `1px solid ${violations.length > 0 ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.1)'}`,
                padding: '2px 10px', borderRadius: 99, fontSize: 11, fontWeight: 700,
              }}>
                {loading ? 'Loading...' : `${violations.length} events`}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <a
                href={`${API}/api/export/csv?camera_id=${camera.camera_id}&date=today`}
                download
                className="btn-primary"
                style={{ fontSize: 11, padding: '4px 12px', textDecoration: 'none', display: 'flex', alignItems: 'center' }}
              >
                <Download size={11} style={{ marginRight: 5 }} />
                Export Today's Logs (CSV)
              </a>
              <button
                className="btn-primary"
                style={{ fontSize: 11, padding: '4px 12px' }}
                onClick={fetchViolations}
                disabled={loading}
              >
                <RefreshCw size={11} className={loading ? 'spin' : ''} style={{ marginRight: 5, display: 'inline' }} />
                Refresh
              </button>
            </div>
          </div>

          {/* Loading bar */}
          {loading && (
            <div style={{
              height: 2, borderRadius: 2, marginBottom: 6, flexShrink: 0,
              background: 'linear-gradient(90deg, #06b6d4, #8b5cf6, #06b6d4)',
              backgroundSize: '200% 100%',
              animation: 'shimmer 1.2s linear infinite',
            }} />
          )}

          {/* Error */}
          {error && (
            <div style={{
              marginBottom: 8, padding: '8px 14px', borderRadius: 8, flexShrink: 0,
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.35)',
              color: '#fca5a5', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span><AlertTriangle size={12} style={{ display: 'inline', marginRight: 6 }} />Failed to load: {error}</span>
              <button
                onClick={fetchViolations}
                style={{ color: '#f87171', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline', fontSize: 12 }}
              >Retry</button>
            </div>
          )}

          {/* Grid */}
          <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 14 }}>
            {!loading && violations.length === 0 && !error ? (
              <div style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', padding: 48, color: 'var(--text-muted)', gap: 10,
              }}>
                <CheckCircle2 size={30} style={{ color: 'var(--accent-emerald)', opacity: 0.6 }} />
                <div style={{ fontSize: 13 }}>No violations logged for this camera yet.</div>
                <button className="btn-primary" style={{ fontSize: 11 }} onClick={fetchViolations}>
                  Force Fetch Now
                </button>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(195px, 1fr))', gap: 10 }}>
                {violations.map(evt => (
                  <div key={evt.event_id} style={{
                    borderRadius: 10, overflow: 'hidden',
                    background: 'rgba(255,255,255,0.03)',
                    border: `1px solid ${evt.status === 'UNRESOLVED' ? 'rgba(239,68,68,0.25)' : 'rgba(255,255,255,0.07)'}`,
                    transition: 'transform 0.15s ease, box-shadow 0.15s ease',
                  }}
                    onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-3px)'; e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.5)'; }}
                    onMouseLeave={e => { e.currentTarget.style.transform = ''; e.currentTarget.style.boxShadow = ''; }}
                  >
                    {/* Snapshot thumbnail */}
                    <div
                      onClick={() => evt.snapshot_url && setSelectedSnap(`${API}${evt.snapshot_url}`)}
                      style={{ position: 'relative', cursor: evt.snapshot_url ? 'zoom-in' : 'default', background: '#0d1117', height: 110, overflow: 'hidden' }}
                    >
                      {evt.snapshot_url ? (
                        <img
                          src={`${API}${evt.snapshot_url}`}
                          alt="Evidence"
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                      ) : (
                        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <AlertTriangle size={22} style={{ color: '#f59e0b' }} />
                        </div>
                      )}
                      {evt.snapshot_url && (
                        <div style={{
                          position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.55)',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          opacity: 0, transition: 'opacity 0.2s',
                        }}
                          onMouseEnter={e => { e.currentTarget.style.opacity = 1; }}
                          onMouseLeave={e => { e.currentTarget.style.opacity = 0; }}
                        >
                          <ZoomIn size={20} color="#fff" />
                        </div>
                      )}
                    </div>

                    {/* Card details */}
                    <div style={{ padding: '8px 10px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{evt.datetime_str}</span>
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: '2px 6px', borderRadius: 99,
                          background: evt.status === 'UNRESOLVED' ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.15)',
                          color: evt.status === 'UNRESOLVED' ? '#f87171' : '#34d399',
                          border: `1px solid ${evt.status === 'UNRESOLVED' ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.3)'}`,
                        }}>
                          {evt.status}
                        </span>
                      </div>

                      <div style={{ fontSize: 11, color: '#fca5a5', fontWeight: 600, marginBottom: 5, lineHeight: 1.35 }}>
                        Missing: {(evt.missing_ppe || []).join(', ') || 'Unknown'}
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{
                          fontSize: 9, fontWeight: 700, padding: '1px 7px', borderRadius: 99,
                          background: `${sevColor(evt.severity)}22`,
                          color: sevColor(evt.severity),
                          border: `1px solid ${sevColor(evt.severity)}44`,
                        }}>
                          {evt.severity || 'HIGH'}
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                          {evt.worker_count || 1} worker{(evt.worker_count || 1) !== 1 ? 's' : ''}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Lightbox */}
      {selectedSnap && (
        <div
          style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            background: 'rgba(0,0,0,0.93)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
          onClick={() => setSelectedSnap(null)}
        >
          <button
            onClick={() => setSelectedSnap(null)}
            style={{
              position: 'absolute', top: 20, right: 24,
              background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: '50%', padding: 8, cursor: 'pointer', color: '#fff',
            }}
          >
            <X size={20} />
          </button>
          <img
            src={selectedSnap}
            alt="Evidence"
            style={{
              maxWidth: '88vw', maxHeight: '86vh', borderRadius: 12,
              boxShadow: '0 30px 90px rgba(0,0,0,0.9)',
            }}
            onClick={e => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
