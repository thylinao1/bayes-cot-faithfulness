/* ============================================================
 * bayes-cot-faithfulness · main.js
 *   - cursor glow tracking (idle-stopping rAF)
 *   - reveal-on-scroll
 *   - nav background on scroll
 *   - smooth in-page navigation
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
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (glow && !reduceMotion && !window.matchMedia('(pointer: coarse)').matches) {
    window.addEventListener('mousemove', (e) => {
      targetX = e.clientX;
      targetY = e.clientY;
      if (!rafId) rafId = window.requestAnimationFrame(animateGlow);
    });
  } else if (glow) {
    glow.style.opacity = '0';
  }

  function animateGlow() {
    curX += (targetX - curX) * 0.12;
    curY += (targetY - curY) * 0.12;
    glow.style.transform = `translate(${curX}px, ${curY}px) translate(-50%, -50%)`;
    // Stop the loop once the glow has settled; the next mousemove restarts it.
    if (Math.abs(targetX - curX) + Math.abs(targetY - curY) < 0.5) {
      rafId = 0;
      return;
    }
    rafId = window.requestAnimationFrame(animateGlow);
  }

  // ---------- reveal-on-scroll ---------------------------------------------
  const revealEls = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window) {
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
  } else {
    revealEls.forEach((el) => el.classList.add('in'));
  }

  // ---------- nav background opacity on scroll ----------------------------
  const nav = document.querySelector('.nav');
  if (nav) {
    window.addEventListener(
      'scroll',
      () => {
        nav.style.background = window.scrollY > 30
          ? 'rgba(7, 9, 15, 0.92)'
          : 'rgba(7, 9, 15, 0.7)';
      },
      { passive: true }
    );
  }

  // ---------- smooth in-page navigation -----------------------------------
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    a.addEventListener('click', (e) => {
      const id = a.getAttribute('href').slice(1);
      if (!id) return;
      const tgt = document.getElementById(id);
      if (!tgt) return;
      e.preventDefault();
      const top = tgt.getBoundingClientRect().top + window.scrollY - 64;
      window.scrollTo({ top, behavior: reduceMotion ? 'auto' : 'smooth' });
    });
  });
})();
