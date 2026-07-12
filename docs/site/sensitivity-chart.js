/* ============================================================
 * sensitivity-chart.js
 *   Native in-page SVG of the rho sensitivity sweep, styled to the
 *   site (no matplotlib). Data is precomputed from the Python package
 *   (notebook 04 config), so the curve is the real validated result.
 *   Interactive: a slider turns the confounding dial over the real
 *   sweep arrays and reads off the faithful / decorative paths live.
 * ============================================================ */

(function () {
  'use strict';

  const DATA = {
    rho: [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    nde: [-0.3686, -0.3238, -0.281, -0.2404, -0.2006, -0.1612, -0.1195, -0.0783, -0.0328, 0.0141, 0.0663, 0.1242, 0.1939, 0.2721, 0.3698],
    nie: [0.6733, 0.6294, 0.5868, 0.546, 0.5068, 0.4679, 0.4273, 0.3862, 0.3408, 0.2942, 0.2417, 0.1843, 0.1151, 0.0335, -0.0636],
    te: [0.3046, 0.3056, 0.3058, 0.3056, 0.3062, 0.3067, 0.3078, 0.3079, 0.308, 0.3083, 0.3081, 0.3085, 0.309, 0.3056, 0.3062],
    true_nde: 0.0909,
    true_nie: 0.2074,
    rho_star: 0.7364,
    rho_true: 0.5,
  };

  const C = {
    cyan: '#4dd0e1', orange: '#ff8c42', green: '#a5e887',
    muted: '#8a93a4', text: '#e6ebf2', text2: '#a4adbd', text3: '#6a7384',
    grid: '#1f2632', edge: '#5a6478', bg: '#0b0f17',
  };

  const W = 720, H = 452;
  const padL = 56, padR = 116, padT = 64, padB = 60;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const xmin = -0.66, xmax = 0.86, ymin = -0.42, ymax = 0.72;

  const sx = (r) => padL + ((r - xmin) / (xmax - xmin)) * innerW;
  const sy = (v) => padT + (1 - (v - ymin) / (ymax - ymin)) * innerH;
  const line = (ys) => DATA.rho.map((r, i) => `${i ? 'L' : 'M'}${sx(r).toFixed(1)},${sy(ys[i]).toFixed(1)}`).join(' ');
  const dots = (ys, color) => DATA.rho
    .map((r, i) => `<circle cx="${sx(r).toFixed(1)}" cy="${sy(ys[i]).toFixed(1)}" r="3" fill="${color}"/>`)
    .join('');
  const mono = "font-family:'JetBrains Mono',monospace";

  // Linear interpolation over the real sweep grid (no extrapolation past ends).
  function interp(ys, rho) {
    const rs = DATA.rho;
    if (rho <= rs[0]) return ys[0];
    if (rho >= rs[rs.length - 1]) return ys[ys.length - 1];
    let i = 0;
    while (i < rs.length - 1 && rs[i + 1] < rho) i++;
    const t = (rho - rs[i]) / (rs[i + 1] - rs[i]);
    return ys[i] + t * (ys[i + 1] - ys[i]);
  }
  const fmt = (x) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(2);

  function gridlines() {
    const yt = [-0.4, -0.2, 0, 0.2, 0.4, 0.6];
    const xt = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8];
    let s = '';
    for (const t of yt) {
      const zero = t === 0;
      s += `<line x1="${padL}" y1="${sy(t)}" x2="${W - padR}" y2="${sy(t)}" stroke="${zero ? C.edge : C.grid}" stroke-width="${zero ? 1 : 0.7}" ${zero ? '' : 'stroke-dasharray="2 5"'}/>`;
      s += `<text x="${padL - 10}" y="${sy(t) + 4}" text-anchor="end" fill="${C.text3}" font-size="12" style="${mono}">${t.toFixed(1)}</text>`;
    }
    for (const t of xt) {
      s += `<text x="${sx(t)}" y="${H - padB + 22}" text-anchor="middle" fill="${C.text3}" font-size="12" style="${mono}">${t.toFixed(1)}</text>`;
    }
    return s;
  }

  function legend() {
    const rows = [
      ['Faithful path', C.cyan, false],
      ['Decorative path', C.orange, false],
      ['Total effect', C.muted, true],
    ];
    const lx = W - padR - 150;
    let s = `<rect x="${lx - 4}" y="${padT - 6}" width="154" height="68" rx="9" fill="${C.bg}" opacity="0.82"/>`;
    rows.forEach(([label, color, dotted], i) => {
      const y = padT + 8 + i * 20;
      s += `<line x1="${lx + 4}" y1="${y}" x2="${lx + 28}" y2="${y}" stroke="${color}" stroke-width="2.4" ${dotted ? 'stroke-dasharray="2 4"' : ''} stroke-linecap="round"/>`;
      s += `<text x="${lx + 36}" y="${y + 4}" fill="${C.text2}" font-size="12">${label}</text>`;
    });
    return s;
  }

  function svgMarkup() {
    const xStar = sx(DATA.rho_star);
    const iTrue = DATA.rho.indexOf(DATA.rho_true);
    return `
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" role="img"
           aria-label="Sensitivity of the faithful path to hidden confounding rho">
        ${gridlines()}

        <!-- ignorability line -->
        <line x1="${sx(0)}" y1="${padT}" x2="${sx(0)}" y2="${H - padB}" stroke="${C.text2}" stroke-width="1" opacity="0.7"/>
        <text x="${sx(0) - 8}" y="${sy(0.66)}" text-anchor="end" fill="${C.text2}" font-size="11" style="${mono}">Assumes</text>
        <text x="${sx(0) - 8}" y="${sy(0.60)}" text-anchor="end" fill="${C.text2}" font-size="11" style="${mono}">ignorability</text>

        <!-- ground-truth guides -->
        <line x1="${padL}" y1="${sy(DATA.true_nie)}" x2="${W - padR}" y2="${sy(DATA.true_nie)}" stroke="${C.cyan}" stroke-width="1" stroke-dasharray="4 4" opacity="0.45"/>
        <line x1="${padL}" y1="${sy(DATA.true_nde)}" x2="${W - padR}" y2="${sy(DATA.true_nde)}" stroke="${C.orange}" stroke-width="1" stroke-dasharray="4 4" opacity="0.45"/>

        <!-- breakdown rho* -->
        <line x1="${xStar}" y1="${padT}" x2="${xStar}" y2="${H - padB}" stroke="${C.cyan}" stroke-width="1" stroke-dasharray="1 4" opacity="0.6"/>
        <text x="${xStar + 6}" y="${sy(0.10)}" fill="${C.cyan}" font-size="11.5" style="${mono}">Breakdown &#961;* = ${DATA.rho_star.toFixed(2)}</text>

        <!-- curves -->
        <path class="sc-curve" d="${line(DATA.te)}" fill="none" stroke="${C.muted}" stroke-width="1.6" stroke-dasharray="2 4"/>
        <path class="sc-curve sc-draw" d="${line(DATA.nde)}" fill="none" stroke="${C.orange}" stroke-width="2.4" stroke-linecap="round"/>
        <path class="sc-curve sc-draw" d="${line(DATA.nie)}" fill="none" stroke="${C.cyan}" stroke-width="2.4" stroke-linecap="round"/>
        ${dots(DATA.nde, C.orange)}${dots(DATA.nie, C.cyan)}

        <!-- recovers-truth marker -->
        <circle cx="${sx(DATA.rho_true)}" cy="${sy(DATA.nie[iTrue])}" r="9" fill="none" stroke="${C.green}" stroke-width="2.4"/>
        <line x1="${sx(DATA.rho_true)}" y1="${sy(DATA.nie[iTrue]) + 11}" x2="${sx(DATA.rho_true)}" y2="${sy(0.01)}" stroke="${C.green}" stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>
        <text x="${sx(DATA.rho_true)}" y="${sy(-0.06)}" text-anchor="middle" fill="${C.green}" font-size="11.5" style="${mono}">True confounding</text>
        <text x="${sx(DATA.rho_true)}" y="${sy(-0.13)}" text-anchor="middle" fill="${C.green}" font-size="11.5" style="${mono}">recovers the truth</text>

        <!-- interactive cursor (moved by the slider) -->
        <g id="sc-cursor" style="pointer-events:none">
          <line id="sc-cursor-line" x1="${sx(0)}" y1="${padT}" x2="${sx(0)}" y2="${H - padB}" stroke="${C.text}" stroke-width="1.4" opacity="0.85"/>
          <circle id="sc-cursor-nie" cx="${sx(0)}" cy="${sy(DATA.nie[6])}" r="6" fill="${C.bg}" stroke="${C.cyan}" stroke-width="2.4"/>
          <circle id="sc-cursor-nde" cx="${sx(0)}" cy="${sy(DATA.nde[6])}" r="6" fill="${C.bg}" stroke="${C.orange}" stroke-width="2.4"/>
        </g>

        ${legend()}

        <!-- axis labels -->
        <text x="${padL + innerW / 2}" y="${H - 8}" text-anchor="middle" fill="${C.text2}" font-size="12.5">Assumed hidden confounding &#961; between the reasoning and the answer</text>
        <text transform="translate(15,${padT + innerH / 2}) rotate(-90)" text-anchor="middle" fill="${C.text2}" font-size="12.5">Effect on the answer</text>
      </svg>`;
  }

  function controlMarkup() {
    return `
      <div class="rho-panel">
        <div class="rho-slider-row">
          <span class="rho-slider-label">Turn the dial: assumed hidden confounding &rho;</span>
          <input type="range" id="rho-slider" min="-0.6" max="0.8" step="0.01" value="0"
                 aria-label="Assumed hidden confounding rho" />
          <output id="rho-val" for="rho-slider">&rho; = +0.00</output>
        </div>
        <div class="rho-readout">
          <div class="rho-metric">
            <span class="rho-k"><span class="rho-swatch cyan"></span>faithful path (NIE)</span>
            <span class="rho-v" id="rho-nie">+0.43</span>
          </div>
          <div class="rho-metric">
            <span class="rho-k"><span class="rho-swatch orange"></span>decorative path (NDE)</span>
            <span class="rho-v" id="rho-nde">&#8722;0.12</span>
          </div>
          <span class="chip chip-clear" id="rho-chip">faithful path holds</span>
        </div>
        <div class="rho-presets">
          <button type="button" class="rho-preset" data-rho="0">assume none (&rho; = 0)</button>
          <button type="button" class="rho-preset" data-rho="0.5">true confounding (&rho; = 0.5)</button>
          <button type="button" class="rho-preset" data-rho="0.7364">breakdown (&rho;* = 0.74)</button>
        </div>
      </div>`;
  }

  function render(el) {
    el.innerHTML = `<div class="sc-figure">${svgMarkup()}</div>${controlMarkup()}`;

    // gentle draw-in (respects reduced motion)
    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      el.querySelectorAll('.sc-draw').forEach((p) => {
        const len = p.getTotalLength();
        p.style.strokeDasharray = len;
        p.style.strokeDashoffset = len;
        p.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.16,1,0.3,1)';
        requestAnimationFrame(() => requestAnimationFrame(() => { p.style.strokeDashoffset = '0'; }));
      });
    }

    // ---- interactivity ----
    const slider = el.querySelector('#rho-slider');
    const cLine = el.querySelector('#sc-cursor-line');
    const cNie = el.querySelector('#sc-cursor-nie');
    const cNde = el.querySelector('#sc-cursor-nde');
    const valEl = el.querySelector('#rho-val');
    const nieEl = el.querySelector('#rho-nie');
    const ndeEl = el.querySelector('#rho-nde');
    const chip = el.querySelector('#rho-chip');

    function update(rho) {
      const x = sx(rho);
      const nie = interp(DATA.nie, rho);
      const nde = interp(DATA.nde, rho);
      cLine.setAttribute('x1', x); cLine.setAttribute('x2', x);
      cNie.setAttribute('cx', x); cNie.setAttribute('cy', sy(nie));
      cNde.setAttribute('cx', x); cNde.setAttribute('cy', sy(nde));
      valEl.innerHTML = '&rho; = ' + fmt(rho);
      nieEl.textContent = fmt(nie);
      ndeEl.textContent = fmt(nde);
      if (nie > 0) {
        chip.textContent = 'faithful path holds';
        chip.className = 'chip chip-clear';
      } else {
        chip.textContent = 'verdict overturned';
        chip.className = 'chip chip-flag';
      }
    }

    slider.addEventListener('input', () => update(parseFloat(slider.value)));
    el.querySelectorAll('.rho-preset').forEach((b) => {
      b.addEventListener('click', () => {
        slider.value = b.dataset.rho;
        update(parseFloat(slider.value));
      });
    });
    update(0);
  }

  function init() {
    const el = document.getElementById('sensitivity-chart');
    if (el) render(el);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
