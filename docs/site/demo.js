/* ============================================================
 * demo.js - "Watch the answer flow" interactive
 *   - Monte Carlo natural-effects (mirrors src/.../effects.py)
 *   - particle flow along the faithful vs shortcut routes
 *   - plain-language readouts, verdict, split bar
 * ============================================================ */

(function () {
  'use strict';

  const byId = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- Monte Carlo natural effects (same model as the Python package) ------
  const N_MC = 4000;
  const STD = new Float32Array(N_MC * 2);
  for (let i = 0; i < STD.length; i++) STD[i] = boxMuller();

  function boxMuller() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  /** Natural direct / indirect / total effect on the probability scale. */
  function effects(alpha, beta, gamma, sigmaM) {
    let p00 = 0, p10 = 0, p11 = 0;
    for (let i = 0; i < N_MC; i++) {
      const m0 = STD[i] * sigmaM;
      const m1 = STD[i + N_MC] * sigmaM + gamma;
      p00 += sigmoid(beta * m0);
      p10 += sigmoid(alpha + beta * m0);
      p11 += sigmoid(alpha + beta * m1);
    }
    p00 /= N_MC; p10 /= N_MC; p11 /= N_MC;
    const nde = p10 - p00;
    const nie = p11 - p10;
    return { nde, nie, te: nde + nie };
  }

  // ---- DOM refs ------------------------------------------------------------
  const S = { alpha: byId('s-alpha'), beta: byId('s-beta'), gamma: byId('s-gamma') };
  const V = { alpha: byId('v-alpha'), beta: byId('v-beta'), gamma: byId('v-gamma') };
  const SIGMA_M = 0.5;
  const SCALE = 0.6; // a plausible max |effect|, for mapping to widths / particle counts

  // ---- particle flow -------------------------------------------------------
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const layer = byId('particles');
  const pathF = byId('routeFaithful');
  const pathS = byId('routeShortcut');
  const lenF = pathF.getTotalLength();
  const lenS = pathS.getTotalLength();
  const MAX_PER_ROUTE = 22;

  function makePool(color) {
    const pool = [];
    for (let i = 0; i < MAX_PER_ROUTE; i++) {
      const c = document.createElementNS(SVG_NS, 'circle');
      c.setAttribute('r', '3.2');
      c.setAttribute('fill', color);
      c.setAttribute('opacity', '0');
      layer.appendChild(c);
      pool.push({ el: c, t: Math.random(), active: false });
    }
    return pool;
  }
  const poolF = makePool('#4dd0e1');
  const poolS = makePool('#ff8c42');
  const SPEED = 0.0017;

  function setActiveCount(pool, n) {
    for (let i = 0; i < pool.length; i++) pool[i].active = i < n;
  }

  function advance(pool, path, len) {
    for (const p of pool) {
      if (!p.active) { p.el.setAttribute('opacity', '0'); continue; }
      p.t += SPEED;
      if (p.t > 1) p.t -= 1;
      const pt = path.getPointAtLength(p.t * len);
      p.el.setAttribute('cx', pt.x.toFixed(1));
      p.el.setAttribute('cy', pt.y.toFixed(1));
      const edge = Math.min(p.t, 1 - p.t);          // fade in/out near the nodes
      p.el.setAttribute('opacity', (0.9 * Math.min(1, edge * 9)).toFixed(2));
    }
  }

  let rafId = 0;
  function loop() {
    advance(poolF, pathF, lenF);
    advance(poolS, pathS, lenS);
    rafId = window.requestAnimationFrame(loop);
  }

  // ---- update from controls -----------------------------------------------
  function update() {
    if (!S.alpha) return;
    const a = parseFloat(S.alpha.value);
    const b = parseFloat(S.beta.value);
    const g = parseFloat(S.gamma.value);
    V.alpha.textContent = a.toFixed(2);
    V.beta.textContent = b.toFixed(2);
    V.gamma.textContent = g.toFixed(2);

    const { nde, nie, te } = effects(a, b, g, SIGMA_M);
    const absTE = Math.abs(te);
    const faith = absTE < 0.005 ? 0.5 : Math.max(0, Math.min(1, nie / te));

    const fMag = Math.max(0, nie);
    const sMag = Math.max(0, nde);
    const fFrac = Math.min(1, fMag / SCALE);
    const sFrac = Math.min(1, sMag / SCALE);

    pathF.setAttribute('stroke-width', (2 + 5.5 * fFrac).toFixed(1));
    pathS.setAttribute('stroke-width', (1.4 + 4.5 * sFrac).toFixed(1));
    if (!reduceMotion) {
      setActiveCount(poolF, Math.round(MAX_PER_ROUTE * fFrac));
      setActiveCount(poolS, Math.round(MAX_PER_ROUTE * sFrac));
    }

    const pct = Math.round(faith * 100);
    byId('gauge-num').textContent = pct + '%';
    byId('live-pct').textContent = pct + '%';

    let vText, vClass, tail;
    if (absTE < 0.01) {
      vText = 'no signal'; vClass = 'v-mixed';
      tail = 'The prompt barely moves the answer at all.';
    } else if (faith >= 0.7) {
      vText = 'faithful reasoning'; vClass = 'v-faithful';
      tail = 'The reasoning is mostly doing the work.';
    } else if (faith >= 0.4) {
      vText = 'mixed'; vClass = 'v-mixed';
      tail = 'The answer is part reasoning, part shortcut.';
    } else {
      vText = 'decorative reasoning'; vClass = 'v-decorative';
      tail = 'The answer mostly arrives by the shortcut.';
    }
    const pill = byId('verdict');
    pill.textContent = vText;
    pill.className = 'verdict-pill ' + vClass;
    byId('live-tail').textContent = tail;

    const total = fMag + sMag || 1;
    const fp = Math.round((100 * fMag) / total);
    const segF = byId('seg-faithful');
    const segS = byId('seg-shortcut');
    segF.style.width = fp + '%';
    segS.style.width = (100 - fp) + '%';
    segF.textContent = fp >= 16 ? 'through reasoning' : '';
    segS.textContent = (100 - fp) >= 16 ? 'shortcut' : '';

    byId('nodeM').setAttribute('stroke-width', (2 + 2.5 * fFrac).toFixed(1));
    byId('nodeY').setAttribute('r', (34 + 3 * Math.min(1, absTE / SCALE)).toFixed(1));
  }

  Object.values(S).forEach((s) => s && s.addEventListener('input', update));

  const PRESETS = {
    faithful: [0.05, 2.4, 1.0],
    decorative: [1.4, 0.3, 0.5],
    mixed: [0.7, 1.2, 0.8],
  };
  document.querySelectorAll('.preset-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const p = PRESETS[btn.dataset.preset];
      if (!p) return;
      S.alpha.value = String(p[0]);
      S.beta.value = String(p[1]);
      S.gamma.value = String(p[2]);
      update();
    });
  });

  // ---- init ----------------------------------------------------------------
  update();
  if (reduceMotion) {
    // a static sprinkle, no animation
    setActiveCount(poolF, Math.min(8, poolF.length));
    setActiveCount(poolS, Math.min(4, poolS.length));
    advance(poolF, pathF, lenF);
    advance(poolS, pathS, lenS);
  } else {
    loop();
  }
})();
