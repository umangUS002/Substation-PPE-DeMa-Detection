// Industrial Substation PPE Operations Dashboard Client Logic

let socket = null;
let soundEnabled = true;
let cameraData = [];
let analyticsData = {};
let complianceChart = null;
let ppeChart = null;
let audioCtx = null;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
    initClock();
    initTabNavigation();
    initFilterHandlers();
    initModalHandlers();
    initSoundToggle();
    initCharts();
    
    // Connect to WebSocket Telemetry Server
    connectWebSocket();
    
    // Initial HTTP API Fetch
    fetchInitialData();
});

// Live Clock
function initClock() {
    const clockEl = document.getElementById("live-clock");
    setInterval(() => {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString() + " (" + Intl.DateTimeFormat().resolvedOptions().timeZone + ")";
    }, 1000);
}

// Sound Synthesizer for Industrial Alarm Beep
function playAlarmSound() {
    if (!soundEnabled) return;
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, audioCtx.currentTime); // High A5 note
        osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.2);
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.linearRampToValueAtTime(0.01, audioCtx.currentTime + 0.25);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + 0.25);
    } catch (e) {
        console.warn("Audio Context sound error:", e);
    }
}

function initSoundToggle() {
    const btn = document.getElementById("btn-toggle-sound");
    btn.addEventListener("click", () => {
        soundEnabled = !soundEnabled;
        btn.textContent = soundEnabled ? "🔊 Audio Alert: ON" : "🔇 Audio Alert: OFF";
        btn.style.background = soundEnabled ? "rgba(255, 171, 0, 0.2)" : "rgba(255, 255, 255, 0.05)";
        btn.style.borderColor = soundEnabled ? "var(--accent-amber)" : "var(--border-color)";
        btn.style.color = soundEnabled ? "var(--accent-amber)" : "var(--text-muted)";
    });
}

// Tab Navigation
function initTabNavigation() {
    const tabs = document.querySelectorAll(".tab-btn");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const targetId = tab.getAttribute("data-tab");
            document.querySelectorAll(".tab-pane").forEach(pane => {
                pane.classList.remove("active");
            });
            document.getElementById(targetId).classList.add("active");

            if (targetId === "tab-incidents") fetchIncidents();
            if (targetId === "tab-management") renderManagementTable();
        });
    });
}

// Filter Handlers
function initFilterHandlers() {
    document.getElementById("filter-zone").addEventListener("change", renderCameraGrid);
    document.getElementById("filter-status").addEventListener("change", renderCameraGrid);
    document.getElementById("search-camera").addEventListener("input", renderCameraGrid);
    document.getElementById("btn-refresh-grid").addEventListener("click", fetchInitialData);
    document.getElementById("btn-refresh-incidents").addEventListener("click", fetchIncidents);
}

// Modal Handlers
function initModalHandlers() {
    const modal = document.getElementById("spotlight-modal");
    document.getElementById("btn-close-modal").addEventListener("click", () => {
        modal.classList.remove("active");
        document.getElementById("modal-stream-img").src = "";
    });
}

function openSpotlight(camId) {
    const cam = cameraData.find(c => c.camera_id === camId);
    if (!cam) return;

    document.getElementById("modal-cam-id").textContent = cam.camera_id;
    document.getElementById("modal-cam-name").textContent = cam.name;
    document.getElementById("modal-zone").textContent = cam.zone;
    document.getElementById("modal-hazard").textContent = cam.hazard_level + " (Helmet, Gloves, Footwear)";
    if (document.getElementById("modal-worker-id")) {
        document.getElementById("modal-worker-id").textContent = `${cam.total_detected_workers || 1} Worker(s) Active`;
    }
    document.getElementById("modal-alert-msg").textContent = cam.alert_message || "All PPE items compliant.";

    const statusBadge = document.getElementById("modal-status-badge");
    if (cam.status === "VIOLATION") {
        statusBadge.className = "status-badge violation";
        statusBadge.textContent = "⚠️ VIOLATION DETECTED";
    } else {
        statusBadge.className = "status-badge compliant";
        statusBadge.textContent = "✅ COMPLIANT";
    }

    document.getElementById("modal-stream-img").src = `/api/camera/${camId}/feed`;
    document.getElementById("spotlight-modal").classList.add("active");
}

// WebSocket Connection
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("[WS] Connected to Industrial Telemetry Gateway.");
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "TELEMETRY_UPDATE" || data.type === "INITIAL_STATE") {
                cameraData = data.cameras || [];
                analyticsData = data.analytics || {};
                updateHeaderKPIs();
                renderCameraGrid();
                updateCharts();
                
                // Check if any camera triggered a new violation sound
                const hasViolation = cameraData.some(c => c.status === "VIOLATION");
                if (hasViolation) playAlarmSound();
            }
        } catch (e) {
            console.error("[WS] Error parsing message:", e);
        }
    };

    socket.onclose = () => {
        console.warn("[WS] Disconnected. Retrying connection in 3 seconds...");
        setTimeout(connectWebSocket, 3000);
    };
}

// Fetch Initial Data via REST API
async function fetchInitialData() {
    try {
        const res = await fetch("/api/cameras");
        const json = await res.json();
        cameraData = json.cameras || [];
        
        const analyticsRes = await fetch("/api/analytics");
        analyticsData = await analyticsRes.json();

        updateHeaderKPIs();
        renderCameraGrid();
        updateCharts();
    } catch (e) {
        console.error("Error fetching camera data:", e);
    }
}

// Update Top Navbar Metrics
function updateHeaderKPIs() {
    const compVal = analyticsData.compliance_rate !== undefined ? analyticsData.compliance_rate : 97.4;
    const topCompEl = document.getElementById("top-compliance-val");
    topCompEl.textContent = compVal + "%";

    const unresolvedAlerts = analyticsData.unresolved_alerts || 0;
    const topAlertsEl = document.getElementById("top-alerts-val");
    topAlertsEl.textContent = unresolvedAlerts;
    document.getElementById("badge-incident-count").textContent = unresolvedAlerts;

    if (document.getElementById("kpi-cams")) {
        document.getElementById("kpi-cams").textContent = `${analyticsData.active_cameras || 24} / ${analyticsData.total_cameras || 24}`;
        document.getElementById("kpi-compliance").textContent = compVal + "%";
        document.getElementById("kpi-scans").textContent = (analyticsData.total_scans || 14280).toLocaleString();
        document.getElementById("kpi-violations").textContent = unresolvedAlerts;
    }
}

// Render 24-Camera Operations Grid
function renderCameraGrid() {
    const container = document.getElementById("camera-grid-container");
    if (!container) return;

    const zoneFilter = document.getElementById("filter-zone").value;
    const statusFilter = document.getElementById("filter-status").value;
    const searchText = document.getElementById("search-camera").value.toLowerCase();

    const filtered = cameraData.filter(cam => {
        if (zoneFilter !== "ALL" && cam.zone !== zoneFilter) return false;
        if (statusFilter !== "ALL" && cam.status !== statusFilter) return false;
        if (searchText && !cam.name.toLowerCase().includes(searchText) && !cam.camera_id.toLowerCase().includes(searchText)) return false;
        return true;
    });

    if (filtered.length === 0) {
        container.innerHTML = `<div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-muted);">No cameras found matching filter criteria.</div>`;
        return;
    }

    container.innerHTML = filtered.map(cam => {
        const isViolation = cam.status === "VIOLATION";
        const statusClass = isViolation ? "violation" : "compliant";
        const statusText = isViolation ? "VIOLATION" : "SAFE";
        const missingList = cam.missing_ppe || [];
        const isLiveWebcam = cam.is_live || cam.camera_id === "CAM-01";

        return `
            <div class="camera-card ${isViolation ? 'violation-active' : ''}" style="${isLiveWebcam ? 'border-color: #00d2ff; box-shadow: 0 0 14px rgba(0, 210, 255, 0.25);' : ''}">
                <div class="camera-header">
                    <div class="cam-title-box">
                        <span class="cam-id" style="${isLiveWebcam ? 'background: #0088ff; color: #fff;' : ''}">${cam.camera_id}</span>
                        <span class="cam-name">${cam.name}</span>
                    </div>
                    <div style="display:flex; gap: 6px; align-items:center;">
                        ${isLiveWebcam ? `<span style="background: rgba(0,210,255,0.15); color: #00d2ff; border: 1px solid #00d2ff; font-size: 10px; font-weight:700; padding: 2px 6px; border-radius: 4px; animation: pulse 2s infinite;">🔴 LIVE WEBCAM (YOLO)</span>` : ''}
                        <span class="status-badge ${statusClass}">${statusText}</span>
                    </div>
                </div>
                
                <div class="camera-viewport">
                    <img class="camera-feed-img" src="/api/camera/${cam.camera_id}/feed" alt="${cam.name}" loading="lazy">
                    <button class="spotlight-btn" onclick="openSpotlight('${cam.camera_id}')">🔍 Spotlight</button>
                </div>

                <div class="camera-footer">
                    <div style="display:flex; justify-content:space-between; color: var(--text-secondary); margin-bottom: 6px;">
                        <span>Zone: <strong>${cam.zone}</strong></span>
                        <span>Feed: <strong style="${isLiveWebcam ? 'color: #00d2ff;' : ''}">${isLiveWebcam ? 'Local Camera (YOLO AI)' : 'Industrial CCTV'}</strong></span>
                    </div>
                    <div class="ppe-tags-list">
                        ${isViolation ? 
                            missingList.map(item => `<span class="ppe-tag missing">❌ NO ${item}</span>`).join('') :
                            `<span class="ppe-tag ok">🪖 Hardhat</span><span class="ppe-tag ok">🦺 Vest</span><span class="ppe-tag ok">🧤 Gloves</span><span class="ppe-tag ok">🥾 Footwear</span>`
                        }
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

// Fetch Incidents Table
async function fetchIncidents() {
    try {
        const res = await fetch("/api/violations?limit=50");
        const data = await res.json();
        const tbody = document.getElementById("incidents-table-body");
        if (!tbody) return;

        if (!data.violations || data.violations.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; padding: 20px; color: var(--text-muted);">No safety violations logged. System fully compliant.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.violations.map(v => `
            <tr>
                <td><strong style="color: var(--accent-cyan);">${v.event_id}</strong></td>
                <td>${v.datetime_str}</td>
                <td>${v.camera_name} (${v.camera_id})</td>
                <td>${v.zone}</td>
                <td><span style="font-size:11px; padding:2px 6px; background:rgba(255,171,0,0.15); color:var(--accent-amber); border-radius:4px;">${v.hazard_level}</span></td>
                <td><strong>${v.worker_id}</strong></td>
                <td><strong style="color: var(--accent-crimson);">Missing: ${v.missing_ppe.join(", ")}</strong></td>
                <td>
                    <span class="status-badge ${v.status === 'UNRESOLVED' ? 'violation' : 'compliant'}">
                        ${v.status}
                    </span>
                </td>
                <td>
                    ${v.status === 'UNRESOLVED' ? 
                        `<button class="action-btn-small" onclick="resolveIncident('${v.event_id}')">Resolve</button>` : 
                        `<span style="color: var(--text-muted); font-size:11px;">Resolved</span>`}
                </td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Error fetching incidents:", e);
    }
}

async function resolveIncident(eventId) {
    try {
        await fetch(`/api/violations/${eventId}/resolve`, { method: "POST" });
        fetchIncidents();
        fetchInitialData();
    } catch (e) {
        console.error("Error resolving incident:", e);
    }
}

// Render Management Table
function renderManagementTable() {
    const tbody = document.getElementById("management-table-body");
    if (!tbody) return;

    tbody.innerHTML = cameraData.map(c => `
        <tr>
            <td><strong style="color: var(--accent-cyan);">${c.camera_id}</strong></td>
            <td>${c.name}</td>
            <td>${c.zone}</td>
            <td><span style="font-size:11px; padding:2px 6px; background:rgba(0,210,255,0.15); color:var(--accent-cyan); border-radius:4px;">${c.hazard_level}</span></td>
            <td><code>rtsp://192.168.10.${c.camera_id.split('-')[1]}/live</code></td>
            <td><span class="status-badge compliant">ONLINE</span></td>
        </tr>
    `).join("");
}

// Initialize Charts
function initCharts() {
    const ctxTrend = document.getElementById("chart-compliance-trend");
    const ctxPpe = document.getElementById("chart-ppe-breakdown");
    if (!ctxTrend || !ctxPpe) return;

    complianceChart = new Chart(ctxTrend, {
        type: 'line',
        data: {
            labels: ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"],
            datasets: [{
                label: 'Safety Compliance Rate (%)',
                data: [94.5, 96.2, 98.0, 97.5, 99.1, 95.8, 93.4, 96.0, 97.8, 98.4, 99.0, 97.4],
                borderColor: '#00e676',
                backgroundColor: 'rgba(0, 230, 118, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { min: 80, max: 100, grid: { color: '#24344d' } },
                x: { grid: { color: '#24344d' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    ppeChart = new Chart(ctxPpe, {
        type: 'bar',
        data: {
            labels: ["Helmet", "Gloves", "Footwear"],
            datasets: [{
                label: 'Missing PPE Incident Count',
                data: [4, 12, 3],
                backgroundColor: ['#ff1744', '#ffab00', '#00d2ff'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: '#24344d' } },
                x: { grid: { color: '#24344d' } }
            }
        }
    });
}

function updateCharts() {
    if (!analyticsData.missing_ppe_breakdown) return;
    if (ppeChart) {
        const counts = [
            analyticsData.missing_ppe_breakdown.Helmet || 0,
            analyticsData.missing_ppe_breakdown.Gloves || 0,
            analyticsData.missing_ppe_breakdown.Footwear || 0
        ];
        ppeChart.data.datasets[0].data = counts;
        ppeChart.update();
    }
}
