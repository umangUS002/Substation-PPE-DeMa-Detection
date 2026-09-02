import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  BarElement, Title, Tooltip, Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { Activity, Shield, ScanSearch, AlertOctagon } from 'lucide-react';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Filler);

const GRID_COLOR = 'rgba(255,255,255,0.05)';
const TICK_COLOR = '#524e72';



function KpiCard({ icon: Icon, iconColor, label, value, sub, subColor, tint }) {
  return (
    <div className={`kpi-card tint-${tint}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="kpi-title">{label}</div>
          <div className="kpi-num" style={subColor ? { color: subColor } : {}}>{value}</div>
        </div>
        <div style={{
          width: 38, height: 38, borderRadius: 9,
          background: `${iconColor}18`, border: `1px solid ${iconColor}25`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0, marginTop: 2,
        }}>
          <Icon size={17} color={iconColor} />
        </div>
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: subColor || 'var(--t3)', marginTop: 6 }}>{sub}</div>
      )}
    </div>
  );
}

export default function AnalyticsDashboard({ analytics }) {
  const compliance     = analytics?.compliance_rate    ?? 97.4;
  const activeCams     = analytics?.active_cameras     || 24;
  const totalCams      = analytics?.total_cameras      || 24;
  const totalScans     = (analytics?.total_scans       || 0).toLocaleString();
  const unresolved     = analytics?.unresolved_alerts  || 0;

  const hourLabels = analytics?.hourly_labels || ['00','02','04','06','08','10','12','14','16','18','20','22'].map(h => `${h}:00`);
  const hourData   = analytics?.hourly_compliance || [94, 96, 98, 97, 99, 96, 93, 96, 98, 98, 99, compliance];

  const ppeBd = analytics?.missing_ppe_breakdown || { Helmet: 4, Footwear: 3 };

  const lineData = {
    labels: hourLabels,
    datasets: [{
      label: 'Compliance %',
      data: hourData,
      borderColor: '#10b981',
      backgroundColor: 'rgba(16,185,129,0.08)',
      pointBackgroundColor: '#10b981',
      pointRadius: 3,
      borderWidth: 2,
      fill: true,
      tension: 0.4,
    }],
  };

  const barData = {
    labels: ['Helmet', 'Footwear'],
    datasets: [{
      label: 'Incidents',
      data: [ppeBd.Helmet || 0, ppeBd.Footwear || 0],
      backgroundColor: ['rgba(244,63,94,0.7)', 'rgba(34,211,238,0.7)'],
      borderRadius: 5,
      borderSkipped: false,
    }],
  };

  const sharedScaleOpts = {
    grid:  { color: GRID_COLOR },
    ticks: { color: TICK_COLOR, font: { size: 11 } },
  };

  const lineOptions = {
    responsive: true,
    animation: { duration: 600 },
    scales: {
      y: { ...sharedScaleOpts, min: 80, max: 100 },
      x: { ...sharedScaleOpts },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(13,17,23,0.9)',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        titleColor: '#f0f4f8',
        bodyColor: '#8899b0',
      },
    },
  };

  const barOptions = {
    responsive: true,
    animation: { duration: 500 },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(13,17,23,0.9)',
        borderColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        titleColor: '#f0f4f8',
        bodyColor: '#8899b0',
      },
    },
    scales: {
      y: { ...sharedScaleOpts, beginAtZero: true },
      x: { ...sharedScaleOpts },
    },
  };

  return (
    <div className="fade-up">
      {/* KPI Row */}
      <div className="analytics-grid">
        <KpiCard
          icon={Activity}        iconColor="#22d3ee" tint="cyan"
          label="Active Cameras" value={`${activeCams} / ${totalCams}`}
          sub="All feeds operational"
        />
        <KpiCard
          icon={Shield}          iconColor="#34d399" tint="emerald"
          label="Compliance Rate" value={`${compliance}%`}
          subColor={compliance >= 90 ? '#34d399' : '#fb7185'}
          sub="PPE standard coverage"
        />
        <KpiCard
          icon={ScanSearch}      iconColor="#a78bfa" tint="violet"
          label="Inspections Today" value={totalScans}
          sub="Continuous frame analysis"
        />
        <KpiCard
          icon={AlertOctagon}    iconColor={unresolved > 0 ? '#fb7185' : '#34d399'} tint="rose"
          label="Open Violations"  value={unresolved}
          subColor={unresolved > 0 ? '#fb7185' : '#34d399'}
          sub={unresolved > 0 ? 'Requires supervisor action' : 'All clear'}
        />
      </div>

      {/* Charts */}
      <div className="charts-row">
        <div className="chart-card">
          <div className="chart-header">24-Hour Compliance Rate (%)</div>
          <Line data={lineData} options={lineOptions} height={90} />
        </div>
        <div className="chart-card">
          <div className="chart-header">Missing PPE Breakdown</div>
          <Bar data={barData} options={barOptions} height={90} />
        </div>
      </div>
    </div>
  );
}
