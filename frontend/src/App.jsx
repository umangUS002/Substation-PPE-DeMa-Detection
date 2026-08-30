import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import TabsNav from './components/TabsNav';
import CameraGrid from './components/CameraGrid';
import SpotlightModal from './components/SpotlightModal';
import IncidentStream from './components/IncidentStream';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import CameraManagement from './components/CameraManagement';
import AuditLog from './components/AuditLog';

export default function App() {
  const [activeTab, setActiveTab] = useState('tab-camera-grid');
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [cameras, setCameras] = useState([]);
  const [analytics, setAnalytics] = useState({});
  const [spotlightCamera, setSpotlightCamera] = useState(null);

  const audioCtxRef = useRef(null);

  const playAlarmSound = () => {
    if (!soundEnabled) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.2);
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.25);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.25);
    } catch (e) {
      console.warn("Audio Context sound error:", e);
    }
  };

  const fetchInitialData = async () => {
    try {
      const res = await fetch('/api/cameras');
      const json = await res.json();
      setCameras(json.cameras || []);

      const analyticsRes = await fetch('/api/analytics');
      const aData = await analyticsRes.json();
      setAnalytics(aData || {});
    } catch (e) {
      console.error("Error fetching initial camera data:", e);
    }
  };

  useEffect(() => {
    fetchInitialData();

    // Setup WebSocket live telemetry connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[WS] Connected to React Live Telemetry.");
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "TELEMETRY_UPDATE" || data.type === "INITIAL_STATE") {
          const cams = data.cameras || [];
          setCameras(cams);
          setAnalytics(data.analytics || {});

          const hasViolation = cams.some(c => c.status === "VIOLATION");
          if (hasViolation) playAlarmSound();
        }
      } catch (e) {
        console.error("Error parsing WS message:", e);
      }
    };

    socket.onclose = () => {
      console.warn("[WS] Disconnected.");
    };

    return () => {
      socket.close();
    };
  }, [soundEnabled]);

  const unresolvedCount = analytics?.unresolved_alerts || 0;

  return (
    <div className="app-shell">
      <Navbar 
        analytics={analytics} 
        soundEnabled={soundEnabled} 
        setSoundEnabled={setSoundEnabled} 
      />

      <TabsNav 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        unresolvedCount={unresolvedCount} 
      />

      <main className="main-container">
        {activeTab === 'tab-camera-grid' && (
          <CameraGrid 
            cameras={cameras} 
            onOpenSpotlight={(cam) => setSpotlightCamera(cam)} 
            onRefresh={fetchInitialData} 
          />
        )}

        {activeTab === 'tab-incidents' && (
          <IncidentStream onIncidentResolved={fetchInitialData} />
        )}

        {activeTab === 'tab-analytics' && (
          <AnalyticsDashboard analytics={analytics} />
        )}

        {activeTab === 'tab-management' && (
          <CameraManagement cameras={cameras} />
        )}

        {activeTab === 'tab-audit' && (
          <AuditLog />
        )}
      </main>

      <SpotlightModal 
        camera={spotlightCamera} 
        onClose={() => setSpotlightCamera(null)} 
      />
    </div>
  );
}
