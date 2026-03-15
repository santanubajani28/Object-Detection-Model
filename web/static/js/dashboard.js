/**
 * dashboard.js — ODM Live Dashboard Client
 * Socket.IO + polling for real-time detection updates
 */

// ── Class colour map ──────────────────────────────────────────────────────────
const CLASS_COLORS = {
    pothole: '#ef4444',
    sign_board: '#f59e0b',
    license_plate: '#3b82f6',
    helmet: '#22c55e',
    no_helmet: '#ef4444',
    wrong_way: '#dc2626',
    vehicle: '#94a3b8',
    person: '#a78bfa',
};

const CLASS_ICONS = {
    pothole: '🕳',
    sign_board: '🪧',
    license_plate: '🔢',
    helmet: '✅',
    no_helmet: '❌',
    wrong_way: '⚠️',
    vehicle: '🚗',
    person: '🚶',
};

function getColor(cls) { return CLASS_COLORS[cls] || '#60a5fa'; }
function getIcon(cls) { return CLASS_ICONS[cls] || '📦'; }

// ── State ─────────────────────────────────────────────────────────────────────
let frameCount = 0;
let lastFrameTime = Date.now();
let fps = 0;
let connected = false;

// ── Socket.IO ─────────────────────────────────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

socket.on('connect', () => {
    connected = true;
    setStatus('Connected', true);
});

socket.on('disconnect', () => {
    connected = false;
    setStatus('Disconnected', false);
});

socket.on('detections', (dets) => {
    updateDetections(dets);
});

// ── Video feed FPS estimator ──────────────────────────────────────────────────
const videoFeed = document.getElementById('videoFeed');
videoFeed.addEventListener('load', () => {
    frameCount++;
    const now = Date.now();
    const elapsed = (now - lastFrameTime) / 1000;
    if (elapsed >= 1.0) {
        fps = (frameCount / elapsed).toFixed(1);
        document.getElementById('fpsValue').textContent = fps;
        frameCount = 0;
        lastFrameTime = now;
    }
    document.getElementById('frameTs').textContent = new Date().toLocaleTimeString();
});

// ── Detection update ──────────────────────────────────────────────────────────
const counters = {};   // cls → count this cycle

function updateDetections(dets) {
    const newCounts = {};
    dets.forEach(d => { newCounts[d.class] = (newCounts[d.class] || 0) + 1; });

    // Update counter tiles
    ['pothole', 'sign_board', 'license_plate', 'helmet', 'no_helmet', 'wrong_way'].forEach(cls => {
        const n = newCounts[cls] || 0;
        const el = document.getElementById(`n-${cls}`);
        const tile = document.getElementById(`cnt-${cls}`);
        if (!el) return;

        const prev = parseInt(el.textContent) || 0;
        el.textContent = n;

        if (n > prev) {
            tile.classList.add('active');
            setTimeout(() => tile.classList.remove('active'), 600);
        }

        // Wrong-way flash the whole screen
        if (cls === 'wrong_way' && n > 0) {
            document.body.classList.add('alert-flash');
            setTimeout(() => document.body.classList.remove('alert-flash'), 400);
        }
    });

    // Total object count
    document.getElementById('detValue').textContent = dets.length;

    // Live detection list
    const list = document.getElementById('detList');
    list.innerHTML = '';
    if (dets.length === 0) {
        list.innerHTML = '<p class="empty-state">No detections this frame</p>';
        return;
    }
    dets.slice(0, 12).forEach(d => {
        const item = document.createElement('div');
        item.className = 'det-item';
        const color = getColor(d.class);
        const icon = getIcon(d.class);
        const label = d.class.replace(/_/g, ' ');
        const conf = (d.confidence * 100).toFixed(0);
        const tid = d.track_id != null ? `#${d.track_id}` : '';
        item.innerHTML = `
      <span class="det-dot" style="background:${color}"></span>
      <span class="det-class">${icon} ${label}</span>
      <span class="det-conf">${conf}%</span>
      <span class="det-id">${tid}</span>
    `;
        list.appendChild(item);
    });
}

// ── Alert log ─────────────────────────────────────────────────────────────────
function refreshAlerts() {
    fetch('/api/alerts')
        .then(r => r.json())
        .then(alerts => {
            const log = document.getElementById('alertLog');
            log.innerHTML = '';
            if (!alerts.length) {
                log.innerHTML = '<p class="empty-state">No alerts yet.</p>';
                return;
            }
            alerts.slice(0, 20).forEach(a => {
                const el = document.createElement('div');
                el.className = `alert-entry alert-${a.severity}`;
                const ts = a.ts ? a.ts.split('T')[1] : '—';
                const cls = (a.class || '').replace(/_/g, ' ');
                const conf = a.confidence ? `${(a.confidence * 100).toFixed(0)}%` : '';
                el.innerHTML = `<span class="alert-ts">${ts}</span><strong>${cls}</strong> ${conf}`;
                log.appendChild(el);
            });
        })
        .catch(() => { });
}

// ── Status helper ─────────────────────────────────────────────────────────────
function setStatus(text, ok) {
    document.getElementById('statusText').textContent = text;
    const dot = document.querySelector('.pulse-dot');
    dot.style.background = ok ? 'var(--success)' : 'var(--danger)';
}

// ── Wrong-way flash CSS (injected dynamically) ─────────────────────────────────
const style = document.createElement('style');
style.textContent = `
  @keyframes flash-red { 0%,100%{box-shadow:none;} 50%{box-shadow:inset 0 0 0 4px rgba(239,68,68,0.6);} }
  body.alert-flash .video-container { animation: flash-red 0.4s ease; }
`;
document.head.appendChild(style);

// ── Auto-refresh alerts every 10 s ────────────────────────────────────────────
refreshAlerts();
setInterval(refreshAlerts, 10000);
