/* ============================================================
 * powermde.js - interactive minimum-detectable-effect calculator
 *   MDE values are precomputed by the repo's own guardrail module
 *   (experiments/07_guardrail_audit.py -> minimum_detectable_rate),
 *   exact binomial, 80% power, one-sided, alpha = 0.05. Embedding
 *   the table means the widget can never drift from the repo.
 * ============================================================ */

(function () {
  'use strict';

  const root = document.getElementById('pmde');
  if (!root) return;

  // [n, MDE percent] from minimum_detectable_rate(n) in the package.
  const TABLE = [
    [10, 14.9], [11, 13.7], [12, 12.6], [14, 10.9], [16, 9.6], [18, 8.6],
    [20, 7.8], [22, 7.1], [25, 6.2], [30, 5.2], [35, 4.5], [40, 4.0],
    [50, 3.2], [60, 2.6], [75, 2.2], [90, 1.8], [103, 1.6], [120, 1.4],
    [150, 1.1], [200, 0.9], [300, 0.5], [400, 0.5], [500, 0.4],
    [750, 0.2], [1000, 0.2],
  ];
  const N_MIN = 10, N_MAX = 1000;
  const TARGET = 10; // guardrail adequacy target: MDE <= 10%

  // log-scale slider position (0..100) <-> n
  const posToN = (p) => Math.round(N_MIN * Math.pow(N_MAX / N_MIN, p / 100));
  const nToPos = (n) => 100 * Math.log(n / N_MIN) / Math.log(N_MAX / N_MIN);

  function mdeAt(n) {
    if (n <= TABLE[0][0]) return TABLE[0][1];
    if (n >= TABLE[TABLE.length - 1][0]) return TABLE[TABLE.length - 1][1];
    let i = 0;
    while (i < TABLE.length - 1 && TABLE[i + 1][0] < n) i++;
    const [n0, m0] = TABLE[i];
    const [n1, m1] = TABLE[i + 1];
    return m0 + (m1 - m0) * ((n - n0) / (n1 - n0));
  }

  const slider = root.querySelector('#pmde-slider');
  const nEl = root.querySelector('#pmde-n');
  const mdeEl = root.querySelector('#pmde-mde');
  const chip = root.querySelector('#pmde-chip');

  function update(n) {
    const mde = mdeAt(n);
    nEl.textContent = 'n = ' + n;
    mdeEl.textContent = mde.toFixed(1) + '%';
    if (mde <= TARGET) {
      chip.textContent = 'adequately powered';
      chip.className = 'chip chip-clear';
    } else {
      chip.textContent = 'underpowered';
      chip.className = 'chip chip-flag';
    }
  }

  slider.addEventListener('input', () => update(posToN(parseFloat(slider.value))));
  root.querySelectorAll('.pmde-preset').forEach((b) => {
    b.addEventListener('click', () => {
      const n = parseInt(b.dataset.n, 10);
      slider.value = String(nToPos(n));
      update(n);
    });
  });

  update(posToN(parseFloat(slider.value)));
})();
