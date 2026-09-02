/* ═══════════════════════════════════════════════════════════════════
   SHARED NEURAL SPACE — DISCOVERY DASHBOARD INTERACTION ENGINE
   ═══════════════════════════════════════════════════════════════════ */

// 1. Concept Vocabulary & Embeddings Simulation
const CONCEPTS = [
  "carousel", "piano", "turntable", "tent", "stove2", "birdbath", "drum", "calf1",
  "mole", "stirrup", "harmonica", "accordion", "backpack", "candle", "canoe",
  "guitar", "trumpet", "violin", "binoculars", "hammer", "umbrella", "toaster"
];

// Pre-computed 2D UMAP/PCA Coordinates for visualization
const CONCEPT_POINTS = [
  { name: "piano", x: 180, y: 120, category: "music", color: "#00F0FF" },
  { name: "drum", x: 260, y: 160, category: "music", color: "#00F0FF" },
  { name: "harmonica", x: 210, y: 90, category: "music", color: "#00F0FF" },
  { name: "accordion", x: 150, y: 160, category: "music", color: "#00F0FF" },
  { name: "guitar", x: 220, y: 200, category: "music", color: "#00F0FF" },
  
  { name: "tent", x: 420, y: 280, category: "outdoor", color: "#9D4EDD" },
  { name: "canoe", x: 470, y: 310, category: "outdoor", color: "#9D4EDD" },
  { name: "backpack", x: 380, y: 250, category: "outdoor", color: "#9D4EDD" },
  { name: "stove2", x: 350, y: 320, category: "outdoor", color: "#9D4EDD" },
  
  { name: "carousel", x: 120, y: 300, category: "object", color: "#FF3366" },
  { name: "birdbath", x: 190, y: 340, category: "object", color: "#FF3366" },
  { name: "candle", x: 250, y: 290, category: "object", color: "#FF3366" },
  { name: "umbrella", x: 160, y: 260, category: "object", color: "#FF3366" },
];

// 2. Tab Navigation
document.addEventListener("DOMContentLoaded", () => {
  initBackgroundCanvas();
  initTabNavigation();
  initManifoldCanvas();
  initInterpolationSlider();
  initMultimodalPlayground();
  initImageryExplorer();
  initScalingCalculator();
});

function initTabNavigation() {
  const btns = document.querySelectorAll(".nav-btn");
  const panels = document.querySelectorAll(".tab-panel");

  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

// 3. Background Neural Network Canvas
function initBackgroundCanvas() {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });

  const particles = Array.from({ length: 45 }, () => ({
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    radius: Math.random() * 2 + 1,
  }));

  function render() {
    ctx.clearRect(0, 0, w, h);

    // Draw connecting lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 140) {
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.15 * (1 - dist / 140)})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Update & draw particles
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;

      ctx.fillStyle = "rgba(0, 240, 255, 0.4)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(render);
  }

  render();
}

// 4. Interactive Manifold Canvas & Geodesic Traversal
let manifoldCtx, manifoldCanvas;
let currentStartConcept = "piano";
let currentEndConcept = "drum";
let currentAlpha = 0.0;

function initManifoldCanvas() {
  manifoldCanvas = document.getElementById("manifold-canvas");
  if (!manifoldCanvas) return;
  manifoldCtx = manifoldCanvas.getContext("2d");

  // Resize canvas to element dimensions
  const rect = manifoldCanvas.parentElement.getBoundingClientRect();
  manifoldCanvas.width = rect.width;
  manifoldCanvas.height = 380;

  drawManifold();
}

function drawManifold() {
  if (!manifoldCtx || !manifoldCanvas) return;
  const ctx = manifoldCtx;
  const w = manifoldCanvas.width;
  const h = manifoldCanvas.height;

  ctx.clearRect(0, 0, w, h);

  // Scale coordinates to canvas width/height
  const scaleX = w / 600;
  const scaleY = h / 400;

  // Find start and end nodes
  const p0 = CONCEPT_POINTS.find(p => p.name === currentStartConcept) || CONCEPT_POINTS[0];
  const p1 = CONCEPT_POINTS.find(p => p.name === currentEndConcept) || CONCEPT_POINTS[1];

  const x0 = p0.x * scaleX;
  const y0 = p0.y * scaleY;
  const x1 = p1.x * scaleX;
  const y1 = p1.y * scaleY;

  // Draw geodesic curve
  ctx.strokeStyle = "rgba(0, 240, 255, 0.4)";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw interpolated current point
  const curX = x0 + (x1 - x0) * currentAlpha;
  const curY = y0 + (y1 - y0) * currentAlpha;

  // Glow on current point
  ctx.shadowColor = "#00F0FF";
  ctx.shadowBlur = 15;
  ctx.fillStyle = "#FFFFFF";
  ctx.beginPath();
  ctx.arc(curX, curY, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Draw concept centroid nodes
  CONCEPT_POINTS.forEach(p => {
    const px = p.x * scaleX;
    const py = p.y * scaleY;

    const isSelected = p.name === currentStartConcept || p.name === currentEndConcept;
    ctx.fillStyle = isSelected ? p.color : "rgba(255, 255, 255, 0.25)";
    ctx.beginPath();
    ctx.arc(px, py, isSelected ? 7 : 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = isSelected ? "#FFFFFF" : "rgba(148, 163, 184, 0.7)";
    ctx.font = isSelected ? "bold 11px Outfit, sans-serif" : "10px Inter, sans-serif";
    ctx.fillText(p.name, px + 9, py + 3);
  });
}

// 5. Interpolation Slider Controller
function initInterpolationSlider() {
  const slider = document.getElementById("interp-slider");
  const alphaVal = document.getElementById("alpha-val");
  const selectStart = document.getElementById("select-concept-a");
  const selectEnd = document.getElementById("select-concept-b");
  const simStartEl = document.getElementById("sim-start-gauge");
  const simEndEl = document.getElementById("sim-end-gauge");
  const simStartTxt = document.getElementById("sim-start-txt");
  const simEndTxt = document.getElementById("sim-end-txt");

  if (selectStart && selectEnd) {
    CONCEPT_POINTS.forEach(p => {
      const optA = document.createElement("option");
      optA.value = p.name;
      optA.textContent = p.name;
      if (p.name === "piano") optA.selected = true;
      selectStart.appendChild(optA);

      const optB = document.createElement("option");
      optB.value = p.name;
      optB.textContent = p.name;
      if (p.name === "drum") optB.selected = true;
      selectEnd.appendChild(optB);
    });

    selectStart.addEventListener("change", (e) => {
      currentStartConcept = e.target.value;
      updateInterpolation();
    });

    selectEnd.addEventListener("change", (e) => {
      currentEndConcept = e.target.value;
      updateInterpolation();
    });
  }

  if (slider) {
    slider.addEventListener("input", (e) => {
      currentAlpha = parseFloat(e.target.value);
      if (alphaVal) alphaVal.textContent = currentAlpha.toFixed(2);
      updateInterpolation();
    });
  }

  function updateInterpolation() {
    drawManifold();

    // Cosine similarity simulation: monotonic transition
    const simA = (1.0 - 0.7 * currentAlpha);
    const simB = (0.3 + 0.7 * currentAlpha);

    if (simStartEl) simStartEl.style.width = `${simA * 100}%`;
    if (simEndEl) simEndEl.style.width = `${simB * 100}%`;
    if (simStartTxt) simStartTxt.textContent = simA.toFixed(3);
    if (simEndTxt) simEndTxt.textContent = simB.toFixed(3);

    // Update candidate ranking
    updateCandidateList(currentAlpha);
  }

  updateInterpolation();
}

function updateCandidateList(alpha) {
  const list = document.getElementById("interp-candidate-list");
  if (!list) return;

  let candidates = [];
  if (alpha <= 0.3) {
    candidates = [
      { name: currentStartConcept, prob: (1.0 - alpha * 1.5) * 0.8 },
      { name: "harmonica", prob: 0.15 },
      { name: currentEndConcept, prob: alpha * 0.6 },
    ];
  } else if (alpha >= 0.7) {
    candidates = [
      { name: currentEndConcept, prob: (0.4 + alpha * 0.5) * 0.8 },
      { name: "guitar", prob: 0.18 },
      { name: currentStartConcept, prob: (1.0 - alpha) * 0.5 },
    ];
  } else {
    candidates = [
      { name: "harmonica (intermediate)", prob: 0.42 },
      { name: currentStartConcept, prob: 0.31 },
      { name: currentEndConcept, prob: 0.27 },
    ];
  }

  list.innerHTML = candidates.map((c, i) => `
    <div class="candidate-item ${i === 0 ? 'top1' : ''}">
      <span class="candidate-rank">#${i + 1}</span>
      <span class="candidate-name">${c.name}</span>
      <span class="candidate-prob">${(c.prob * 100).toFixed(1)}%</span>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width: ${c.prob * 100}%"></div>
      </div>
    </div>
  `).join("");
}

// 6. Multimodal Retrieval & Single-Trial EEG Playback
function initMultimodalPlayground() {
  const canvas = document.getElementById("eeg-wave-canvas");
  const playBtn = document.getElementById("play-eeg-btn");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  canvas.width = canvas.parentElement.getBoundingClientRect().width;
  canvas.height = 140;

  let offset = 0;
  let isPlaying = true;

  function drawEEGWave() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const midY = canvas.height / 2;

    // Draw 3 simulated EEG channels
    const channels = [
      { color: "#00F0FF", freq: 0.04, amp: 22, phase: 0 },
      { color: "#9D4EDD", freq: 0.06, amp: 18, phase: 1.5 },
      { color: "#FF3366", freq: 0.08, amp: 14, phase: 3.0 },
    ];

    channels.forEach(ch => {
      ctx.strokeStyle = ch.color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();

      for (let x = 0; x < canvas.width; x++) {
        const y = midY + Math.sin((x + offset) * ch.freq + ch.phase) * ch.amp
                        + Math.sin((x + offset) * 0.12) * (ch.amp * 0.4);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    if (isPlaying) {
      offset += 1.5;
    }
    requestAnimationFrame(drawEEGWave);
  }

  drawEEGWave();

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      isPlaying = !isPlaying;
      playBtn.textContent = isPlaying ? "⏸ Pause Stream" : "▶ Resume Stream";
    });
  }
}

// 7. Perception vs Mental Imagery Explorer
function initImageryExplorer() {
  const slider = document.getElementById("imagery-time-slider");
  const timeVal = document.getElementById("imagery-time-val");
  const deltaVal = document.getElementById("imagery-delta-val");
  const regionBadge = document.getElementById("imagery-region-badge");

  if (!slider) return;

  slider.addEventListener("input", (e) => {
    const tMs = parseInt(e.target.value);
    if (timeVal) timeVal.textContent = `${tMs > 0 ? '+' : ''}${tMs} ms`;

    // Calculate simulated reinstatement delta S(t)
    // Peak at 440 ms (0.3767)
    let delta = 0.0;
    if (tMs < 100) {
      delta = 0.02 * Math.sin(tMs * 0.02);
    } else {
      delta = 0.38 * Math.exp(-Math.pow((tMs - 440) / 140, 2));
    }
    if (deltaVal) deltaVal.textContent = `ΔS = ${delta.toFixed(4)}`;

    if (regionBadge) {
      if (tMs < 200) {
        regionBadge.textContent = "Primary Occipital Sensory (Bottom-Up Perception)";
        regionBadge.style.color = "#00F0FF";
      } else if (tMs >= 350 && tMs <= 550) {
        regionBadge.textContent = "Parietal-Frontal Top-Down Reinstatement Peak";
        regionBadge.style.color = "#FF3366";
      } else {
        regionBadge.textContent = "Higher-Order Associative Integration";
        regionBadge.style.color = "#9D4EDD";
      }
    }
  });
}

// 8. Population Scaling Calculator
function initScalingCalculator() {
  const slider = document.getElementById("scaling-n-slider");
  const nVal = document.getElementById("scaling-n-val");
  const projAcc = document.getElementById("scaling-proj-acc");
  const projSnr = document.getElementById("scaling-proj-snr");

  if (!slider) return;

  slider.addEventListener("input", (e) => {
    const n = parseInt(e.target.value);
    if (nVal) nVal.textContent = `${n} Subjects`;

    // Power-law: Acc(N) = A_inf - beta * N^(-0.5)
    // Base parameters from Phase 10
    const A_inf = 0.085;
    const beta = 0.065;
    const gamma = 0.50;
    const acc = Math.max(0.02, A_inf - beta * Math.pow(n, -gamma));
    const snrGain = Math.sqrt(n);

    if (projAcc) projAcc.textContent = `${(acc * 100).toFixed(2)}%`;
    if (projSnr) projSnr.textContent = `${snrGain.toFixed(2)}x Gain`;
  });
}
