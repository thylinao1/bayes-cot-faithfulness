/* ============================================================
 * sensitivity-chart.js
 *   Native in-page SVG of the rho sensitivity sweep, styled to the
 *   site (no matplotlib). Data is precomputed from the Python package
 *   (notebook 04 config), so the curve is the real validated result.
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
    muted: '#8a93a4', text2: '#a4adbd', text3: '#6a7384',
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

  function gridlines() {
    const yt = [-0.4, -0.2, 0, 0.2, 0.4, 0.6];
    const xt = [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8];
    let s = '';
    for (const t of yt) {
      const zero = t === 0;
      s += `<line x1="${padL}" y1="${sy(t)}" x2="${W - padR}" y2="${sy(t)}" stroke="${zero ? C.edge : C.grid}" stroke-width="${zero ? 1 : 0.7}" ${zero ? '' : 'stroke-dasharray="2 5"'}/>`;
      s += `<text x="${padL - 10}" y="${sy(t) + 4}" text-anchor="end" fill="${C.text3}" font-size="11" style="${mono}">${t.toFixed(1)}</text>`;
    }
    for (const t of xt) {
      s += `<text x="${sx(t)}" y="${H - padB + 22}" text-anchor="middle" fill="${C.text3}" font-size="11" style="${mono}">${t.toFixed(1)}</text>`;
    }
    return s;
  }

  function legend() {
    const rows = [
      ['Faithful path', C.cyan, false],
      ['Decorative path', C.orange, false],
      ['Total effect', C.muted, true],
    ];
    let s = `<rect x="${padL - 2}" y="${padT - 6}" width="160" height="68" rx="9" fill="${C.bg}" opacity="0.66"/>`;
    rows.forEach(([label, color, dotted], i) => {
      const y = padT + 8 + i * 20;
      s += `<line x1="${padL + 8}" y1="${y}" x2="${padL + 32}" y2="${y}" stroke="${color}" stroke-width="2.4" ${dotted ? 'stroke-dasharray="2 4"' : ''} stroke-linecap="round"/>`;
      s += `<text x="${padL + 40}" y="${y + 4}" fill="${C.text2}" font-size="12">${label}</text>`;
    });
    return s;
  }

  function render(el) {
    const xStar = sx(DATA.rho_star);
    const yStarTrue = sy(DATA.true_nie);
    const iTrue = DATA.rho.indexOf(DATA.rho_true);
    const svg = `
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" role="img"
           aria-label="Sensitivity of the faithful path to hidden confounding rho">
        ${gridlines()}

        <!-- ignorability line -->
        <line x1="${sx(0)}" y1="${padT}" x2="${sx(0)}" y2="${H - padB}" stroke="${C.text2}" stroke-width="1" opacity="0.7"/>
        <text x="${sx(0) - 8}" y="${padT + 46}" text-anchor="end" fill="${C.text2}" font-size="11" style="${mono}">Assumes</text>
        <text x="${sx(0) - 8}" y="${padT + 60}" text-anchor="end" fill="${C.text2}" font-size="11" style="${mono}">ignorability</text>

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
        <text x="${sx(DATA.rho_true) + 16}" y="${sy(DATA.nie[iTrue]) - 14}" fill="${C.green}" font-size="11.5" style="${mono}">True confounding</text>
        <text x="${sx(DATA.rho_true) + 16}" y="${sy(DATA.nie[iTrue]) - 1}" fill="${C.green}" font-size="11.5" style="${mono}">recovers the truth</text>

        ${legend()}

        <!-- axis labels -->
        <text x="${padL + innerW / 2}" y="${H - 8}" text-anchor="middle" fill="${C.text2}" font-size="12.5">Assumed hidden confounding &#961; between the reasoning and the answer</text>
        <text transform="translate(15,${padT + innerH / 2}) rotate(-90)" text-anchor="middle" fill="${C.text2}" font-size="12.5">Effect on the answer</text>
      </svg>`;
    el.innerHTML = svg;

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
