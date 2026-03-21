/* ================================================================
   MineSafe-AI — JavaScript Application Logic
   ================================================================ */

document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initNavbar();
  initScrollReveal();
  initSliders();
  initPredictor();
  loadDashboardData();
});

/* ==========================  PARTICLES  ========================== */
function initParticles() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let particles = [];
  let animId;

  function resize() {
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.size = Math.random() * 2 + 0.5;
      this.speedX = (Math.random() - 0.5) * 0.5;
      this.speedY = (Math.random() - 0.5) * 0.5;
      this.opacity = Math.random() * 0.5 + 0.1;
    }
    update() {
      this.x += this.speedX;
      this.y += this.speedY;
      if (this.x < 0 || this.x > canvas.width) this.speedX *= -1;
      if (this.y < 0 || this.y > canvas.height) this.speedY *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245, 158, 11, ${this.opacity})`;
      ctx.fill();
    }
  }

  const count = Math.min(80, Math.floor((canvas.width * canvas.height) / 12000));
  for (let i = 0; i < count; i++) particles.push(new Particle());

  function connectParticles() {
    for (let a = 0; a < particles.length; a++) {
      for (let b = a + 1; b < particles.length; b++) {
        const dx = particles[a].x - particles[b].x;
        const dy = particles[a].y - particles[b].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          ctx.beginPath();
          ctx.moveTo(particles[a].x, particles[a].y);
          ctx.lineTo(particles[b].x, particles[b].y);
          ctx.strokeStyle = `rgba(245, 158, 11, ${0.06 * (1 - dist / 130)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => { p.update(); p.draw(); });
    connectParticles();
    animId = requestAnimationFrame(animate);
  }
  animate();
}

/* ==========================  NAVBAR  ========================== */
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  const toggle = document.querySelector('.nav-toggle');
  const links = document.querySelector('.nav-links');

  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 60);
  });

  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
    });
    links.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => links.classList.remove('open'));
    });
  }
}

/* ==========================  SCROLL REVEAL  ========================== */
function initScrollReveal() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
  );
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

/* ==========================  SLIDERS  ========================== */
function initSliders() {
  document.querySelectorAll('.param-slider').forEach(slider => {
    const valueEl = document.getElementById(slider.id + '_val');
    if (valueEl) {
      valueEl.textContent = slider.value;
      slider.addEventListener('input', () => {
        valueEl.textContent = slider.value;
      });
    }
  });
}

/* ==========================  PREDICTOR  ========================== */
function initPredictor() {
  const form = document.getElementById('predictorForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = form.querySelector('.predict-btn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing…';

    const params = {};
    form.querySelectorAll('.param-slider').forEach(s => {
      params[s.name] = parseFloat(s.value);
    });

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      console.error('Prediction error:', err);
      document.getElementById('resultContent').innerHTML =
        '<p style="color:var(--red);">Prediction failed. Please try again.</p>';
    } finally {
      btn.disabled = false;
      btn.innerHTML = '⚡ Analyze Risk';
    }
  });
}

function renderResults(data) {
  const placeholder = document.getElementById('resultPlaceholder');
  const content = document.getElementById('resultContent');
  if (placeholder) placeholder.style.display = 'none';
  if (!content) return;
  content.style.display = 'block';

  // Risk score & gauge
  const score = data.risk_score || 0;
  const riskLevel = data.risk_level || data.predicted_class || 'Unknown';
  const confidence = data.confidence || 0;

  // Gauge animation
  const circumference = 2 * Math.PI * 90;
  const offset = circumference - (score / 100) * circumference;
  const gaugeFill = document.querySelector('.gauge-fill');
  if (gaugeFill) {
    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = offset;
    // colour
    if (score >= 65) gaugeFill.style.stroke = '#ef4444';
    else if (score >= 35) gaugeFill.style.stroke = '#f59e0b';
    else gaugeFill.style.stroke = '#10b981';
  }

  // Value
  const gaugeVal = document.getElementById('gaugeValue');
  if (gaugeVal) animateCounter(gaugeVal, 0, Math.round(score), 1200);

  // Risk level badge
  const levelEl = document.getElementById('riskLevelBadge');
  if (levelEl) {
    levelEl.textContent = riskLevel;
    levelEl.className = 'result-risk-level ';
    if (riskLevel.includes('High')) levelEl.classList.add('risk-high');
    else if (riskLevel.includes('Medium')) levelEl.classList.add('risk-medium');
    else levelEl.classList.add('risk-low');
  }

  // Confidence
  const confEl = document.getElementById('confidenceText');
  if (confEl) confEl.textContent = `Model confidence: ${(confidence * 100).toFixed(1)}%`;

  // Recommendations
  const recList = document.getElementById('recommendationList');
  if (recList && data.recommendations) {
    recList.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
  }
}

/* ==========================  DASHBOARD  ========================== */
async function loadDashboardData() {
  try {
    const res = await fetch('/api/stats');
    const stats = await res.json();
    renderMetrics(stats);
    renderCharts(stats);
  } catch (err) {
    console.error('Stats load error:', err);
  }
}

function renderMetrics(stats) {
  const els = {
    metricAccuracy: stats.model_performance?.accuracy,
    metricAUC: stats.model_performance?.auc_score,
    metricSamples: stats.total_samples,
    metricFeatures: stats.total_features,
  };

  Object.entries(els).forEach(([id, value]) => {
    const el = document.getElementById(id);
    if (!el || value == null) return;
    if (typeof value === 'number' && value < 1) {
      animateCounter(el, 0, Math.round(value * 100), 1500, '%');
    } else {
      animateCounter(el, 0, value, 1500);
    }
  });
}

function renderCharts(stats) {
  // Risk Distribution Doughnut
  if (stats.risk_distribution) {
    const ctx1 = document.getElementById('riskDistChart')?.getContext('2d');
    if (ctx1) {
      const labels = Object.keys(stats.risk_distribution);
      const values = Object.values(stats.risk_distribution);
      const colors = labels.map(l => {
        if (l.toLowerCase().includes('high')) return '#ef4444';
        if (l.toLowerCase().includes('medium')) return '#f59e0b';
        return '#10b981';
      });
      new Chart(ctx1, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{ data: values, backgroundColor: colors, borderColor: '#111827', borderWidth: 3 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          cutout: '65%',
          plugins: {
            legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { family: 'Inter' } } },
          },
        },
      });
    }
  }

  // Feature Importance Bar
  if (stats.feature_importance) {
    const ctx2 = document.getElementById('featureChart')?.getContext('2d');
    if (ctx2) {
      const labels = stats.feature_importance.features.slice(0, 10).map(f =>
        f.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      );
      const values = stats.feature_importance.values.slice(0, 10);
      new Chart(ctx2, {
        type: 'bar',
        data: {
          labels,
          datasets: [{
            data: values,
            backgroundColor: values.map((_, i) => `rgba(245, 158, 11, ${1 - i * 0.08})`),
            borderRadius: 6,
            barThickness: 20,
          }],
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: true,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#64748b', font: { family: 'Inter' } }, grid: { color: 'rgba(255,255,255,0.04)' } },
            y: { ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }, grid: { display: false } },
          },
        },
      });
    }
  }

  // Radar — Regional
  const ctx3 = document.getElementById('regionalChart')?.getContext('2d');
  if (ctx3) {
    new Chart(ctx3, {
      type: 'radar',
      data: {
        labels: ['Rainfall', 'Slope Risk', 'Seismic', 'Weathering', 'Blast Impact'],
        datasets: [
          { label: 'Jharkhand', data: [70, 85, 60, 75, 55], borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.15)', pointBackgroundColor: '#ef4444' },
          { label: 'Odisha', data: [80, 65, 50, 60, 45], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.15)', pointBackgroundColor: '#f59e0b' },
          { label: 'Rajasthan', data: [30, 45, 40, 35, 60], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.15)', pointBackgroundColor: '#10b981' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: { display: false },
            grid: { color: 'rgba(255,255,255,0.06)' },
            angleLines: { color: 'rgba(255,255,255,0.06)' },
            pointLabels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
          },
        },
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 16, font: { family: 'Inter' } } },
        },
      },
    });
  }

  // Performance bar
  const ctx4 = document.getElementById('perfChart')?.getContext('2d');
  if (ctx4 && stats.model_performance) {
    const perf = stats.model_performance;
    new Chart(ctx4, {
      type: 'bar',
      data: {
        labels: ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1'],
        datasets: [{
          data: [perf.accuracy, perf.auc_score, perf.precision, perf.recall, perf.f1_score],
          backgroundColor: ['#f59e0b', '#3b82f6', '#8b5cf6', '#10b981', '#ec4899'],
          borderRadius: 8,
          barThickness: 36,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 1, ticks: { color: '#64748b', callback: v => (v * 100) + '%' }, grid: { color: 'rgba(255,255,255,0.04)' } },
          x: { ticks: { color: '#94a3b8', font: { family: 'Inter' } }, grid: { display: false } },
        },
      },
    });
  }
}

/* ==========================  COUNTER ANIMATION  ========================== */
function animateCounter(el, start, end, duration, suffix = '') {
  const startTime = performance.now();
  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (end - start) * eased);
    el.textContent = current + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
