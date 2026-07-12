/* ============================================================
 * scrolly.js - "Watch the auditor catch it"
 *   Scroll-driven walkthrough of a real caught transcript.
 *   Degrades to a fully visible static transcript with no JS.
 * ============================================================ */

(function () {
  'use strict';

  const root = document.getElementById('scrolly');
  if (!root) return;

  const transcript = document.getElementById('scrolly-transcript');
  const steps = Array.from(root.querySelectorAll('.step-card'));
  if (!transcript || !steps.length) return;

  // Mark JS-enhanced so CSS can dim inactive lines and stage the verdict.
  root.classList.add('scrolly-js');

  const lines = Array.from(transcript.querySelectorAll('[data-line]'));

  function setStep(n) {
    transcript.setAttribute('data-active', String(n));
    steps.forEach((s) => s.classList.toggle('active', +s.dataset.step === n));
    lines.forEach((el) => el.classList.toggle('lit', +el.dataset.line === n));
  }

  setStep(1);

  if (!('IntersectionObserver' in window)) {
    // No observer: reveal the final state so nothing is hidden.
    setStep(4);
    return;
  }

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          setStep(+entry.target.dataset.step);
        }
      });
    },
    // Activate the card crossing the vertical middle of the viewport.
    { rootMargin: '-45% 0px -45% 0px', threshold: 0 }
  );
  steps.forEach((s) => io.observe(s));
})();
