const state = {
  stream: { algorithm: "AI-phase2", events: [] },
  index: 0,
  history: [],
  lastFrameAt: 0,
  frameMs: 1800,
  liveMode: false,
  livePollId: null,
};

const LIVE_STREAM_PATH = "/api/aiops-stream.json?limit=500";
const STATIC_STREAM_PATH = "aiops-stream.json";
const SAMPLE_STREAM_PATH = "sample-aiops-stream.json";
const LIVE_POLL_MS = 2000;

const els = {
  tick: document.getElementById("current-tick"),
  riskLevel: document.getElementById("risk-level"),
  riskScore: document.getElementById("risk-score"),
  alertCount: document.getElementById("alert-count"),
  algorithm: document.getElementById("algorithm-label"),
  serverGrid: document.getElementById("server-grid"),
  riskCanvas: document.getElementById("risk-canvas"),
  utilizationCanvas: document.getElementById("utilization-canvas"),
  chartRiskLabel: document.getElementById("chart-risk-label"),
  aiopsLevelBadge: document.getElementById("aiops-level-badge"),
  rootCause: document.getElementById("root-cause"),
  tags: document.getElementById("aiops-tags"),
  recommendations: document.getElementById("recommendations"),
  guardrails: document.getElementById("guardrails"),
  events: document.getElementById("event-stream"),
};

async function loadStream() {
  const loaded = await loadFirstAvailable([
    LIVE_STREAM_PATH,
    STATIC_STREAM_PATH,
    SAMPLE_STREAM_PATH,
  ]);
  if (loaded) {
    state.liveMode = loaded.path === LIVE_STREAM_PATH;
    replaceStream(loaded.data, { jumpToLatest: state.liveMode });
  } else {
    state.stream = fallbackStream();
    els.algorithm.textContent = state.stream.algorithm || "AI-phase2";
    renderFrame(0);
  }
  if (state.liveMode) startLivePolling();
  requestAnimationFrame(loop);
}

async function loadFirstAvailable(paths) {
  for (const path of paths) {
    try {
      return { path, data: await loadJson(path) };
    } catch (error) {
      console.warn(`stream unavailable: ${path}`, error);
    }
  }
  return null;
}

function replaceStream(stream, { jumpToLatest = false } = {}) {
  state.stream = stream;
  els.algorithm.textContent = stream.algorithm || "AI-phase2";
  state.index = jumpToLatest ? Math.max(stream.events.length - 1, 0) : Math.min(state.index, stream.events.length - 1);
  renderFrame(state.index);
}

function startLivePolling() {
  if (state.livePollId) return;
  state.livePollId = window.setInterval(pollLiveStream, LIVE_POLL_MS);
}

async function pollLiveStream() {
  try {
    const nextStream = await loadJson(LIVE_STREAM_PATH);
    const previousLatest = state.stream.events[state.stream.events.length - 1];
    const nextLatest = nextStream.events[nextStream.events.length - 1];
    if (!previousLatest || previousLatest.tick !== nextLatest.tick) {
      replaceStream(nextStream, { jumpToLatest: true });
    }
  } catch (error) {
    console.warn("live stream polling failed", error);
  }
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`stream load failed: ${response.status}`);
  const data = await response.json();
  if (!data.events || data.events.length === 0) {
    throw new Error(`stream has no events: ${path}`);
  }
  return data;
}

function loop(timestamp) {
  if (!state.lastFrameAt) state.lastFrameAt = timestamp;
  if (timestamp - state.lastFrameAt > state.frameMs) {
    state.index = (state.index + 1) % state.stream.events.length;
    renderFrame(state.index);
    state.lastFrameAt = timestamp;
  }
  requestAnimationFrame(loop);
}

function renderFrame(index) {
  const event = state.stream.events[index];
  if (!event) return;
  state.history.push(event);
  if (state.history.length > 24) state.history.shift();

  renderTopline(event);
  renderServerGrid(event.servers);
  renderAIOpsPanel(event);
  drawRiskChart(state.history);
  drawUtilizationChart(state.history);
}

function renderTopline(event) {
  const { aiops } = event;
  els.tick.textContent = event.tick;
  els.riskLevel.textContent = aiops.risk_level;
  els.riskScore.textContent = formatScore(aiops.risk_score);
  els.alertCount.textContent = aiops.active_alerts.length;
  els.chartRiskLabel.textContent = formatScore(aiops.risk_score);
}

function renderServerGrid(servers) {
  els.serverGrid.replaceChildren(
    ...servers.map((server) => {
      const card = document.createElement("article");
      card.className = `server-card ${server.status}`;
      card.title = `Server ${server.id}: CPU ${server.cpu}%, MEM ${server.mem}%, NET ${server.net}%`;
      card.innerHTML = `
        <div class="server-head">
          <span>Server ${server.id}</span>
          <span class="server-state">${server.status}</span>
        </div>
        ${metric("CPU", server.cpu, "cpu")}
        ${metric("MEM", server.mem, "mem")}
        ${metric("NET", server.net, "net")}
      `;
      return card;
    }),
  );
}

function metric(label, value, kind) {
  return `
    <div class="metric">
      <div class="metric-row"><span>${label}</span><span>${value}%</span></div>
      <div class="bar ${kind}"><span style="width:${Math.min(value, 100)}%"></span></div>
    </div>
  `;
}

function renderAIOpsPanel(event) {
  const { aiops } = event;
  els.aiopsLevelBadge.className = `risk-badge ${aiops.risk_level}`;
  els.aiopsLevelBadge.textContent = aiops.risk_level;
  els.rootCause.textContent = aiops.root_cause_summary;

  els.tags.replaceChildren(...renderTags(aiops.risk_tags));
  els.recommendations.replaceChildren(...renderRecommendations(aiops.recommendations));
  els.guardrails.replaceChildren(...renderGuardrails(aiops.guardrails));
  els.events.replaceChildren(...renderEvents(event.events, event.tick));
}

function renderTags(tags) {
  const values = tags.length ? tags : ["no-active-risk"];
  return values.map((tag) => {
    const item = document.createElement("span");
    item.className = "tag";
    item.textContent = tag;
    return item;
  });
}

function renderRecommendations(recommendations) {
  const values = recommendations.length
    ? recommendations
    : [{ action: "no-new-recommendation", reason: "No advisory action emitted for this tick.", expected_effect: "" }];
  return values.map((rec) => {
    const item = document.createElement("article");
    item.className = "recommendation";
    item.innerHTML = `
      <strong>${rec.action}</strong>
      <p>${rec.reason || ""}</p>
      <p>${rec.expected_effect || ""}</p>
    `;
    return item;
  });
}

function renderGuardrails(guardrails) {
  return guardrails.map((guardrail) => {
    const item = document.createElement("article");
    item.className = "guardrail";
    item.innerHTML = `<strong>${guardrail}</strong><p>Policy changes stay advisory unless approved outside the realtime path.</p>`;
    return item;
  });
}

function renderEvents(events, tick) {
  return events.map((eventText) => {
    const item = document.createElement("li");
    item.textContent = `t=${tick}: ${eventText}`;
    return item;
  });
}

function drawRiskChart(history) {
  const points = history.map((item) => item.aiops.risk_score);
  drawLineChart(els.riskCanvas, [
    { values: points, color: "#c2413a", label: "Risk" },
  ], 1);
}

function drawUtilizationChart(history) {
  const cpu = history.map((item) => item.global_state.active_cpu_util);
  const mem = history.map((item) => item.global_state.active_mem_util);
  const net = history.map((item) => item.global_state.active_net_util);
  drawLineChart(
    els.utilizationCanvas,
    [
      { values: cpu, color: "#20895d", label: "CPU" },
      { values: mem, color: "#2f68c4", label: "MEM" },
      { values: net, color: "#15839a", label: "NET" },
    ],
    1,
  );
}

function drawLineChart(canvas, series, maxY) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = 28;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f6f8fa";
  ctx.fillRect(0, 0, width, height);
  drawGrid(ctx, width, height, pad);

  for (const line of series) {
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 3;
    line.values.forEach((value, index) => {
      const x = pad + (index / Math.max(line.values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - (value / maxY) * (height - pad * 2);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function drawGrid(ctx, width, height, pad) {
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i += 1) {
    const y = pad + i * ((height - pad * 2) / 4);
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#657286";
  ctx.font = "12px system-ui";
  ctx.fillText("1.0", 4, pad + 4);
  ctx.fillText("0.0", 4, height - pad + 4);
}

function formatScore(value) {
  return Number(value || 0).toFixed(2);
}

function fallbackStream() {
  const ticks = [1, 8, 15, 23, 30, 40];
  const risks = [
    ["low", 0, [], "No elevated AIOps risk signals detected."],
    ["low", 0.1, ["migration-watch"], "Elevated AIOps signals detected: migration-watch."],
    ["medium", 0.3, ["network-watch", "migration-watch"], "Network utilization is approaching the watch threshold."],
    ["critical", 0.95, ["network-pressure", "migration-pressure", "sla-risk"], "Network pressure is the dominant SLA risk signal."],
    ["high", 0.55, ["network-pressure", "migration-watch", "sla-risk"], "Network pressure remains active while recommendation cooldown is in effect."],
    ["low", 0.1, ["migration-watch"], "Network risk has cleared; migration watch remains low."],
  ];
  return {
    algorithm: "AI-phase2",
    events: ticks.map((tick, index) => demoEvent(tick, risks[index], index)),
  };
}

function demoEvent(tick, [level, score, tags, cause], index) {
  const netBase = [18, 44, 66, 86, 82, 57][index];
  const recommendations =
    level === "critical"
      ? [
          {
            action: "enable-network-headroom-protection",
            reason: "Network pressure and SLA risk are both active.",
            expected_effect: "Preserve network headroom.",
          },
        ]
      : [];
  return {
    tick,
    global_state: {
      active_cpu_util: [0.31, 0.46, 0.58, 0.67, 0.63, 0.52][index],
      active_mem_util: [0.28, 0.42, 0.55, 0.62, 0.61, 0.49][index],
      active_net_util: [0.24, 0.61, 0.78, 0.91, 0.87, 0.63][index],
    },
    aiops: {
      risk_level: level,
      risk_score: score,
      risk_tags: tags,
      active_alerts: tags.map((tag) => ({ tag, occurrence_count: 1 })),
      root_cause_summary: cause,
      recommendations,
      guardrails: ["do_not_auto_apply", "human_approval_required"],
    },
    servers: Array.from({ length: 8 }, (_, id) => ({
      id,
      status: netBase + id * 2 > 90 ? "overload" : netBase + id * 2 > 72 ? "warning" : "normal",
      cpu: Math.min(95, 20 + index * 9 + id * 3),
      mem: Math.min(95, 18 + index * 8 + id * 2),
      net: Math.min(98, netBase + id * 2),
    })),
    events: [`AIOps ${level} state at tick ${tick}.`],
  };
}

loadStream();
