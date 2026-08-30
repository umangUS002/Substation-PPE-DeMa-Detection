import React, { useState } from 'react';
import { Search, RefreshCw, Maximize2, Users } from 'lucide-react';

const ZONES = [
  'All Zones',
  'HV Switchyard',
  'Transformer Yard',
  'Control Building',
  'Switchgear Hall',
  'Maintenance Yard',
  'Plant Entrance & Gate',
  'Solar Array Field',
];

export default function CameraGrid({ cameras, onOpenSpotlight, onRefresh }) {
  const [zone,   setZone]   = useState('All Zones');
  const [status, setStatus] = useState('ALL');
  const [query,  setQuery]  = useState('');

  const filtered = cameras.filter(cam => {
    if (zone !== 'All Zones' && cam.zone !== zone) return false;
    if (status === 'VIOLATION'  && cam.status !== 'VIOLATION')  return false;
    if (status === 'COMPLIANT'  && cam.status !== 'COMPLIANT')  return false;
    if (query && !cam.name.toLowerCase().includes(query.toLowerCase()) &&
        !cam.camera_id.toLowerCase().includes(query.toLowerCase())) return false;
    return true;
  });

  const violationCount = cameras.filter(c => c.status === 'VIOLATION').length;

  return (
    <div className="fade-up">
      {/* ── FILTER BAR ── */}
      <div className="toolbar-bar">
        <div className="filter-group">
          {/* Violation summary pill */}
          {violationCount > 0 && (
            <div style={{
              background: 'var(--red-dim)', border: '1px solid rgba(244,63,94,0.3)',
              color: 'var(--red)', fontSize: 12, fontWeight: 700,
              padding: '5px 12px', borderRadius: 99, display: 'flex', alignItems: 'center', gap: 6,
            }}>
              ⚠ {violationCount} Active Violation{violationCount !== 1 ? 's' : ''}
            </div>
          )}
          <span className="count-tag">{filtered.length} feeds</span>
        </div>

        <div className="filter-group">
          {/* Search */}
          <div style={{ position: 'relative' }}>
            <Search size={13} style={{
              position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)',
              color: 'var(--text-3)', pointerEvents: 'none',
            }} />
            <input
              className="search-input"
              style={{ paddingLeft: 30, width: 180 }}
              placeholder="Search camera…"
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
          </div>

          {/* Zone filter */}
          <select className="filter-select" value={zone} onChange={e => setZone(e.target.value)}>
            {ZONES.map(z => <option key={z} value={z}>{z}</option>)}
          </select>

          {/* Status filter */}
          <select className="filter-select" value={status} onChange={e => setStatus(e.target.value)}>
            <option value="ALL">All Status</option>
            <option value="VIOLATION">⚠ Violations</option>
            <option value="COMPLIANT">✅ Compliant</option>
          </select>

          <button className="btn-primary" onClick={onRefresh}>
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>

      {/* ── GRID ── */}
      <div className="camera-grid">
        {filtered.length === 0 ? (
          <div style={{
            gridColumn: '1 / -1', padding: '60px 0',
            textAlign: 'center', color: 'var(--text-3)', fontSize: 13,
          }}>
            No cameras match the selected filters.
          </div>
        ) : (
          filtered.map(cam => {
            const isViol  = cam.status === 'VIOLATION';
            const isLive  = cam.is_live || cam.camera_id === 'CAM-01';
            const workers = cam.total_detected_workers || 1;
            const missing = cam.missing_ppe || [];

            return (
              <div
                key={cam.camera_id}
                className={`camera-card${isViol ? ' violation-active' : ''}${isLive ? ' live-cam-card' : ''}`}
              >
                {/* Header */}
                <div className="camera-header">
                  <div className="cam-title-box">
                    <span className={`cam-id${isLive ? ' live-id' : ''}`}>{cam.camera_id}</span>
                    <span className="cam-name">{cam.name}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0 }}>
                    {isLive && <span className="live-badge-mini">● LIVE</span>}
                    <span className={`status-badge ${isViol ? 'violation' : 'compliant'}`}>
                      {isViol ? 'ALERT' : 'SAFE'}
                    </span>
                  </div>
                </div>

                {/* Feed */}
                <div className="camera-viewport">
                  <img
                    className="camera-feed-img"
                    src={`/api/camera/${cam.camera_id}/feed`}
                    alt={cam.name}
                    loading="lazy"
                  />
                  <button className="spotlight-btn" onClick={() => onOpenSpotlight(cam)}>
                    <Maximize2 size={12} /> Spotlight & Logs
                  </button>
                </div>

                {/* Footer */}
                <div className="camera-footer">
                  <div className="cam-meta-row">
                    <span className="cam-zone-label">
                      <strong>{cam.zone}</strong>
                    </span>
                    <span className="cam-worker-label">
                      <Users size={11} /> {workers} {workers === 1 ? 'worker' : 'workers'}
                    </span>
                  </div>

                  <div className="ppe-tags-list">
                    {isViol ? (
                      missing.length > 0 ? (
                        missing.map((item, i) => (
                          <span key={i} className="ppe-tag missing">✕ {item}</span>
                        ))
                      ) : (
                        <span className="ppe-tag missing">PPE Violation</span>
                      )
                    ) : (
                      <>
                        <span className="ppe-tag ok">Helmet ✓</span>
                        <span className="ppe-tag ok">Gloves ✓</span>
                        <span className="ppe-tag ok">Boots ✓</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
