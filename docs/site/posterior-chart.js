/* ============================================================
 * posterior-chart.js
 *   Native in-page SVG of the three posterior-recovery histograms,
 *   styled to the site (no matplotlib). Bin counts are precomputed
 *   from the PyMC posterior (notebook 02), so the shapes are real.
 * ============================================================ */

(function () {
  'use strict';

  const D = {
    nde: { edges: [-0.0889, -0.0797, -0.0706, -0.0615, -0.0524, -0.0432, -0.0341, -0.025, -0.0158, -0.0067, 0.0024, 0.0115, 0.0207, 0.0298, 0.0389, 0.0481, 0.0572, 0.0663, 0.0755, 0.0846, 0.0937, 0.1028, 0.112, 0.1211, 0.1302, 0.1394, 0.1485, 0.1576, 0.1667, 0.1759, 0.185, 0.1941, 0.2033, 0.2124, 0.2215, 0.2306, 0.2398, 0.2489, 0.258, 0.2672, 0.2763], counts: [2, 5, 6, 18, 31, 37, 59, 73, 144, 162, 215, 288, 308, 356, 436, 496, 503, 538, 572, 515, 484, 463, 474, 381, 321, 289, 218, 159, 141, 111, 72, 47, 24, 18, 12, 13, 5, 2, 1, 1], truth: 0.0662, lo: -0.0194, hi: 0.1843 },
    nie: { edges: [0.0727, 0.0798, 0.0869, 0.094, 0.1011, 0.1082, 0.1153, 0.1224, 0.1295, 0.1366, 0.1437, 0.1508, 0.1579, 0.165, 0.1721, 0.1792, 0.1863, 0.1934, 0.2005, 0.2076, 0.2147, 0.2218, 0.2289, 0.236, 0.2431, 0.2502, 0.2573, 0.2644, 0.2715, 0.2786, 0.2857, 0.2929, 0.3, 0.3071, 0.3142, 0.3213, 0.3284, 0.3355, 0.3426, 0.3497, 0.3568], counts: [1, 2, 2, 6, 10, 13, 25, 42, 54, 74, 134, 167, 237, 278, 314, 400, 442, 490, 597, 556, 584, 570, 540, 462, 453, 372, 317, 258, 163, 153, 111, 71, 44, 23, 18, 7, 6, 3, 0, 1], truth: 0.2277, lo: 0.1408, hi: 0.2911 },
    te: { edges: [0.1791, 0.1845, 0.1899, 0.1953, 0.2007, 0.2061, 0.2115, 0.2169, 0.2223, 0.2277, 0.2331, 0.2385, 0.2439, 0.2493, 0.2547, 0.26, 0.2654, 0.2708, 0.2762, 0.2816, 0.287, 0.2924, 0.2978, 0.3032, 0.3086, 0.314, 0.3194, 0.3248, 0.3302, 0.3356, 0.341, 0.3464, 0.3518, 0.3572, 0.3625, 0.3679, 0.3733, 0.3787, 0.3841, 0.3895, 0.3949], counts: [1, 0, 1, 5, 1, 10, 11, 18, 40, 47, 74, 95, 131, 164, 207, 252, 352, 388, 486, 517, 522, 559, 628, 582, 536, 491, 452, 382, 326, 256, 173, 117, 61, 55, 26, 17, 10, 5, 1, 1], truth: 0.2939, lo: 0.2378, hi: 0.3503 },
  };

  const PANELS = [
    ['nde', '#ff8c42', 'Natural Direct Effect', 'the shortcut'],
    ['nie', '#4dd0e1', 'Natural Indirect Effect', 'the faithful path'],
    ['te', '#a5e887', 'Total Effect', 'both paths'],
  ];
  const TEXT = '#a4adbd', TEXT3 = '#6a7384', GRID = '#1f2632', WHITE = '#e6ebf2';
  const W = 320, H = 250, padL = 30, padR = 16, padT = 50, padB = 38;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const mono = "font-family:'JetBrains Mono',monospace";

  function panel(key, color, title, subtitle) {
    const d = D[key];
    const e0 = d.edges[0], eN = d.edges[d.edges.length - 1];
    const ymax = Math.max(...d.counts) * 1.1;
    const x = (v) => padL + ((v - e0) / (eN - e0)) * innerW;
    const y = (c) => padT + (1 - c / ymax) * innerH;
    const baseY = padT + innerH;

    let bars = '';
    for (let i = 0; i < d.counts.length; i++) {
      const bx = x(d.edges[i]);
      const bw = Math.max(0.5, x(d.edges[i + 1]) - bx - 0.5);
      const by = y(d.counts[i]);
      const center = ((d.edges[i] + d.edges[i + 1]) / 2).toFixed(3);
      bars += `<rect class="hbar hbar-${key}" data-v="${center}" data-c="${d.counts[i]}" x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${(baseY - by).toFixed(1)}" fill="${color}" fill-opacity="0.5" stroke="${color}" stroke-width="0.5"/>`;
    }

    // x ticks (4, rounded to 2dp)
    let xticks = '';
    for (let t = 0; t <= 3; t++) {
      const v = e0 + (t / 3) * (eN - e0);
      xticks += `<text x="${x(v).toFixed(1)}" y="${baseY + 18}" text-anchor="middle" fill="${TEXT3}" font-size="10" style="${mono}">${v.toFixed(2)}</text>`;
    }

    const guide = (v, stroke, dash) =>
      `<line x1="${x(v).toFixed(1)}" y1="${padT}" x2="${x(v).toFixed(1)}" y2="${baseY}" stroke="${stroke}" stroke-width="${dash ? 1.1 : 1.6}" ${dash ? 'stroke-dasharray="4 3"' : ''} opacity="${dash ? 0.85 : 1}"/>`;

    return `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Posterior recovery of the three natural effects on synthetic data; all three 95% credible intervals contain the truth" style="width:100%;height:auto;display:block">
        <line x1="${padL}" y1="${baseY}" x2="${W - padR}" y2="${baseY}" stroke="${GRID}" stroke-width="1"/>
        ${bars}
        ${guide(d.lo, color, true)}${guide(d.hi, color, true)}
        ${guide(d.truth, WHITE, false)}
        ${xticks}
        <text x="${padL}" y="22" fill="${WHITE}" font-size="13" font-weight="600" font-family="Inter">${title}</text>
        <text x="${padL}" y="38" fill="${color}" font-size="11" style="${mono}">truth = ${d.truth >= 0 ? '+' : ''}${d.truth.toFixed(3)} &middot; ${subtitle}</text>
        <text id="readout-${key}" x="${W - padR}" y="22" text-anchor="end" fill="${color}" font-size="11" style="${mono}"></text>
      </svg>`;
  }

  function render(el) {
    const panels = PANELS
      .map(([k, c, t, s]) => `<div style="flex:1;min-width:250px">${panel(k, c, t, s)}</div>`)
      .join('');
    el.innerHTML =
      `<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end">${panels}</div>` +
      `<div style="text-align:center;margin-top:6px;font-size:11px;color:${TEXT3};${mono}">` +
      'hover a bar &middot; white line: ground truth &middot; dashed: 95% credible interval</div>';

    // tasteful hover: bars brighten (CSS) and the panel shows the bin value + draw count
    PANELS.forEach(([key]) => {
      const readout = el.querySelector('#readout-' + key);
      if (!readout) return;
      el.querySelectorAll('.hbar-' + key).forEach((bar) => {
        bar.addEventListener('mouseenter', () => {
          readout.textContent = '≈ ' + bar.dataset.v + ' · ' + bar.dataset.c + ' draws';
        });
        bar.addEventListener('mouseleave', () => { readout.textContent = ''; });
      });
    });
  }

  function init() {
    const el = document.getElementById('posterior-chart');
    if (el) render(el);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
