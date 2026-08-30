import React from 'react';
import { Camera, AlertTriangle, BarChart2, Settings2, ClipboardList } from 'lucide-react';

const TABS = [
  { id: 'tab-camera-grid', label: 'Live Cameras', icon: Camera },
  { id: 'tab-incidents',   label: 'Incidents',    icon: AlertTriangle, hasBadge: true },
  { id: 'tab-analytics',  label: 'Analytics',    icon: BarChart2 },
  { id: 'tab-management', label: 'Cameras',       icon: Settings2 },
  { id: 'tab-audit',      label: 'Audit Log',     icon: ClipboardList },
];

export default function TabsNav({ activeTab, setActiveTab, unresolvedCount }) {
  return (
    <nav className="nav-tabs">
      {TABS.map(({ id, label, icon: Icon, hasBadge }) => {
        const active = activeTab === id;
        return (
          <button
            key={id}
            className={`tab-btn${active ? ' active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={14} />
            {label}
            {hasBadge && unresolvedCount > 0 && (
              <span className="tab-badge">{unresolvedCount}</span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
