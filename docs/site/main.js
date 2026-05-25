/* ============================================================
 * bayes-cot-faithfulness · main.js
 *   - cursor glow tracking
 *   - reveal-on-scroll
 *   - interactive Monte Carlo demo (alpha, beta, gamma, sigma)
 * ============================================================ */

(function () {
  'use strict';

  // ---------- cursor glow ---------------------------------------------------
  const glow = document.getElementById('cursorGlow');
  let rafId = 0;
  let targetX = window.innerWidth / 2;
  let targetY = window.innerHeight / 2;
  let curX = targetX;
  let curY = targetY;

  if (glow && !window.matchMedia('(pointer: coarse)').matches) {
    window.addEventListener('mousemove', (e) => {
      targetX = e.clientX;
      targetY = e.clientY;
      if (!rafId) animateGlow();
    });
  } else if (glow) {
    glow.style.opacity = '0';
  }

  function animateGlow() {
    curX += (targetX - curX) * 0.12;
    curY += (targetY - curY) * 0.12;
    glow.style.transform = `translate(${curX}px, ${curY}px) translate(-50%, -50%)`;
    rafId = window.requestAnimationFrame(animateGlow);
  }

  // ---------- reveal-on-scroll ---------------------------------------------
  const revealEls = document.querySelectorAll('.reveal');
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('in');
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: '0px 0px -50px 0px' }
  );
  revealEls.forEach((el) => io.observe(el));

  // ---------- nav background opacity on scroll ----------------------------
  const nav = document.querySelector('.nav');
  if (nav) {
    window.addEventListener(
      'scroll',
      () => {
        const y = window.scrollY;
        nav.style.background = y > 30
          ? 'rgba(7, 9, 15, 0.92)'
          : 'rgba(7, 9, 15, 0.7)';
      },
      { passive: true }
    );
  }

  // ============================================================
  // Interactive Monte Carlo demo
  // ============================================================

  const N_MC = 5000;            // samples per estimate
  const RNG = mulberry32(0xc0ffee);

  // Pre-draw standard normals once for performance; rescale on the fly.
  const STD_NORMALS = new Float32Array(N_MC * 2);
  for (let i = 0; i < STD_NORMALS.length; i++) STD_NORMALS[i] = boxMuller();

  function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
      t = (t + 0x6d2b79f5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  function boxMuller() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  function sigmoid(z) {
    return 1.0 / (1.0 + Math.exp(-z));
  }

  /**
   * Compute (NDE, NIE, TE) on the probability scale by Monte Carlo
   * integration over the mediator distribution, matching the Python
   * implementation in src/bayes_cot_faithfulness/effects.py.
   */
  function computeEffects(alpha, beta, gamma, sigmaM) {
    // m_under_x0 ~ Normal(0, sigma_m)
    // m_under_x1 ~ Normal(gamma, sigma_m)
    let p_x0_m0 = 0;
    let p_x1_m0 = 0;
    let p_x1_m1 = 0;

    for (let i = 0; i < N_MC; i++) {
      const z0 = STD_NORMALS[i] * sigmaM;
      const z1 = STD_NORMALS[i + N_MC] * sigmaM + gamma;
      p_x0_m0 += sigmoid(alpha * 0 + beta * z0);
      p_x1_m0 += sigmoid(alpha * 1 + beta * z0);
      p_x1_m1 += sigmoid(alpha * 1 + beta * z1);
    }
    p_x0_m0 /= N_MC;
    p_x1_m0 /= N_MC;
    p_x1_m1 /= N_MC;

    const nde = p_x1_m0 - p_x0_m0;
    const nie = p_x1_m1 - p_x1_m0;
    const te = nde + nie;
    return { nde, nie, te };
  }

  // ----- DOM refs ---------------------------------------------------------
  const sliders = {
    alpha: document.getElementById('alphaSlider'),
    beta: document.getElementById('betaSlider'),
    gamma: document.getElementById('gammaSlider'),
    sigma: document.getElementById('sigmaSlider'),
  };
  const labels = {
    alpha: document.getElementById('alphaValue'),
    beta: document.getElementById('betaValue'),
    gamma: document.getElementById('gammaValue'),
    sigma: document.getElementById('sigmaValue'),
  };
  const bars = {
    nde: document.getElementById('ndeBar'),
    nie: document.getElementById('nieBar'),
    te: document.getElementById('teBar'),
  };
  const values = {
    nde: document.getElementById('ndeValue'),
    nie: document.getElementById('nieValue'),
    te: document.getElementById('teValue'),
  };
  const meterFill = document.getElementById('meterFill');
  const meterValue = document.getElementById('meterValue');
  const meterVerdict = document.getElementById('meterVerdict');

  function fmt(x) {
    const s = x >= 0 ? '+' : '-';
    return s + Math.abs(x).toFixed(2);
  }

  function updateDemo() {
    if (!sliders.alpha) return;
    const a = parseFloat(sliders.alpha.value);
    const b = parseFloat(sliders.beta.value);
    const g = parseFloat(sliders.gamma.value);
    const s = parseFloat(sliders.sigma.value);

    labels.alpha.textContent = a.toFixed(2);
    labels.beta.textContent = b.toFixed(2);
    labels.gamma.textContent = g.toFixed(2);
    labels.sigma.textContent = s.toFixed(2);

    const eff = computeEffects(a, b, g, s);

    // Width is |effect| relative to a fixed scale (max plausible |TE| ~0.6).
    const SCALE = 0.6;
    bars.nde.style.width = Math.min(100, (Math.abs(eff.nde) / SCALE) * 100) + '%';
    bars.nie.style.width = Math.min(100, (Math.abs(eff.nie) / SCALE) * 100) + '%';
    bars.te.style.width = Math.min(100, (Math.abs(eff.te) / SCALE) * 100) + '%';
    values.nde.textContent = fmt(eff.nde);
    values.nie.textContent = fmt(eff.nie);
    values.te.textContent = fmt(eff.te);

    // Faithfulness = NIE share of total absolute effect (signed mediator path
    // relative to total). When TE is near zero we report 50 % (indeterminate).
    let faithfulness;
    const absTE = Math.abs(eff.te);
    if (absTE < 0.005) {
      faithfulness = 0.5;
    } else {
      const ratio = eff.nie / eff.te;
      faithfulness = Math.max(0, Math.min(1, ratio));
    }

    meterFill.style.width = (faithfulness * 100).toFixed(0) + '%';
    meterValue.textContent = (faithfulness * 100).toFixed(0) + ' %';

    let verdictText, verdictClass;
    if (absTE < 0.01) {
      verdictText = 'no signal';
      verdictClass = 'verdict';
    } else if (faithfulness >= 0.7) {
      verdictText = 'faithful reasoning';
      verdictClass = 'verdict cyan-text';
    } else if (faithfulness >= 0.4) {
      verdictText = 'mixed';
      verdictClass = 'verdict mid-text';
    } else {
      verdictText = 'decorative CoT';
      verdictClass = 'verdict orange-text';
    }
    meterVerdict.textContent = verdictText;
    meterVerdict.className = verdictClass;
  }

  Object.values(sliders).forEach((sl) => {
    if (sl) sl.addEventListener('input', updateDemo);
  });

  function applyPreset(a, b, g, s) {
    if (!sliders.alpha) return;
    sliders.alpha.value = a;
    sliders.beta.value = b;
    sliders.gamma.value = g;
    sliders.sigma.value = s;
    updateDemo();
  }
  const pFaithful = document.getElementById('presetFaithful');
  const pDecor = document.getElementById('presetDecorative');
  const pMixed = document.getElementById('presetMixed');
  if (pFaithful) pFaithful.addEventListener('click', () => applyPreset(0.05, 2.0, 1.0, 0.4));
  if (pDecor) pDecor.addEventListener('click', () => applyPreset(1.5, 0.3, 0.4, 0.6));
  if (pMixed) pMixed.addEventListener('click', () => applyPreset(0.7, 1.2, 0.7, 0.5));

  updateDemo();

  // ============================================================
  // Smooth in-page navigation
  // ============================================================
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      if (!id) return;
      const tgt = document.getElementById(id);
      if (!tgt) return;
      e.preventDefault();
      const top = tgt.getBoundingClientRect().top + window.scrollY - 64;
      window.scrollTo({ top, behavior: 'smooth' });
    });
  });
})();
