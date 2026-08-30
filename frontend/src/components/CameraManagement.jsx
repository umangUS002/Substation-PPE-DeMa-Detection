import React from 'react';
import { Plus } from 'lucide-react';

export default function CameraManagement({ cameras }) {
  return (
    <div>
      <div className="toolbar-bar">
        <h3 style={{ color: 'var(--text-primary)' }}>Plant Industrial Camera Registry</h3>
        <button className="btn-primary" onClick={() => alert('Camera auto-discovery active for 24 channels.')}>
          <Plus size={14} /> Add RTSP Stream
        </button>
      </div>

      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Camera ID</th>
              <th>Name</th>
              <th>Plant Zone</th>
              <th>Hazard Protocol</th>
              <th>Stream Source</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {cameras.map(c => (
              <tr key={c.camera_id}>
                <td><strong style={{ color: 'var(--accent-cyan)' }}>{c.camera_id}</strong></td>
                <td>{c.name}</td>
                <td>{c.zone}</td>
                <td>
                  <span style={{ fontSize: 11, padding: '2px 6px', background: 'rgba(0,210,255,0.15)', color: 'var(--accent-cyan)', borderRadius: 4 }}>
                    {c.hazard_level}
                  </span>
                </td>
                <td><code>rtsp://192.168.10.{c.camera_id.split('-')[1]}/live</code></td>
                <td><span className="status-badge compliant">ONLINE</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
