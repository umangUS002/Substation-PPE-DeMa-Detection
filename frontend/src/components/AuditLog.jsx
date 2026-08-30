import React from 'react';
import { Download, ShieldCheck, FileSpreadsheet } from 'lucide-react';

export default function AuditLog() {
  const zoneAudits = [
    { zone: 'HV Switchyard', cameras: 6, scans: '14,210', compliance: '98.4%', violations: 2, status: 'NOMINAL' },
    { zone: 'Transformer Yard', cameras: 4, scans: '9,840', compliance: '97.8%', violations: 3, status: 'MONITORED' },
    { zone: 'Control Building', cameras: 3, scans: '8,450', compliance: '99.5%', violations: 0, status: 'EXCELLENT' },
    { zone: 'Switchgear Hall', cameras: 2, scans: '5,200', compliance: '98.9%', violations: 1, status: 'NOMINAL' },
    { zone: 'Maintenance Yard', cameras: 3, scans: '6,780', compliance: '96.2%', violations: 4, status: 'ATTENTION' },
    { zone: 'Plant Entrance & Gate', cameras: 2, scans: '7,150', compliance: '99.1%', violations: 1, status: 'EXCELLENT' },
    { zone: 'Solar Array Field', cameras: 2, scans: '3,890', compliance: '100.0%', violations: 0, status: 'EXCELLENT' },
  ];

  return (
    <div>
      <div className="toolbar-bar">
        <div>
          <h3 style={{ margin: 0, fontSize: 16, color: 'var(--text-primary)' }}>
            Plant Substation Zone Safety & Compliance Audit
          </h3>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            Consolidated inspection metrics across 24 monitored substation channels
          </span>
        </div>
        <a href="/api/export/csv" download="substation_safety_audit.csv" className="btn-primary" style={{ textDecoration: 'none' }}>
          <Download size={14} /> Export Audit CSV
        </a>
      </div>

      <div className="data-table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Substation Zone</th>
              <th>Monitored Cameras</th>
              <th>Total Inspection Scans</th>
              <th>Safety Compliance Rate</th>
              <th>Recorded Violations</th>
              <th>Operational Status</th>
            </tr>
          </thead>
          <tbody>
            {zoneAudits.map((z, idx) => (
              <tr key={idx}>
                <td><strong style={{ color: 'var(--accent-cyan)' }}>{z.zone}</strong></td>
                <td>{z.cameras} Channels</td>
                <td>{z.scans}</td>
                <td><strong style={{ color: 'var(--accent-emerald)' }}>{z.compliance}</strong></td>
                <td>{z.violations}</td>
                <td>
                  <span className={`status-badge ${z.status === 'ATTENTION' ? 'violation' : 'compliant'}`}>
                    {z.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
