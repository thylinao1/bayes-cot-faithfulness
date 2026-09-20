/* ui.js — page chrome: reveals, glossary popups, nav state, reading
   progress, scrollytelling, and the hero driver.

   The hero driver watches its own frame time and steps quality down
   rather than letting a fanless laptop thermally throttle the whole page. */
(function () {
  "use strict";

  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------- reveal */
  (function reveals() {
    var els = document.querySelectorAll(".reveal");
    if (!els.length) return;
    if (reduced || !("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -9% 0px", threshold: 0.05 });
    els.forEach(function (el) { obs.observe(el); });
  })();

  /* ------------------------------------------------- glossary popups */
  (function glossary() {
    var defs = document.getElementById("defs");
    var scrim = document.getElementById("scrim");
    var modal = document.getElementById("modal");
    if (!defs || !scrim || !modal) return;
    var body = modal.querySelector(".modal-body");
    var closeBtn = modal.querySelector(".modal-x");
    var lastFocus = null;

    function open(id) {
      var src = defs.querySelector("#" + id);
      if (!src) return;
      lastFocus = document.activeElement;
      body.innerHTML = src.innerHTML;
      var h = body.querySelector("h3");
      if (h) { h.id = "modal-title"; modal.setAttribute("aria-labelledby", "modal-title"); }
      scrim.hidden = false; modal.hidden = false;
      requestAnimationFrame(function () {
        scrim.classList.add("on"); modal.classList.add("on");
        closeBtn.focus();
      });
      document.documentElement.style.overflow = "hidden";
    }

    function close() {
      scrim.classList.remove("on"); modal.classList.remove("on");
      document.documentElement.style.overflow = "";
      setTimeout(function () {
        scrim.hidden = true; modal.hidden = true; body.innerHTML = "";
      }, 320);
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }

    document.querySelectorAll(".term").forEach(function (b) {
      b.addEventListener("click", function () { open(b.dataset.modal); });
    });
    scrim.addEventListener("click", close);
    closeBtn.addEventListener("click", close);
    addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modal.classList.contains("on")) close();
    });
    /* keep tab focus inside the dialog while it is open */
    modal.addEventListener("keydown", function (e) {
      if (e.key !== "Tab") return;
      var f = modal.querySelectorAll("button, a[href]");
      if (!f.length) return;
      var first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    });
  })();

  /* --------------------------------------- nav state + read progress */
  (function chrome() {
    var nav = document.querySelector(".nav");
    var bar = document.querySelector(".progress");
    var links = [].slice.call(document.querySelectorAll(".nav-links a[href^='#']"));
    var targets = links.map(function (a) {
      return document.querySelector(a.getAttribute("href"));
    });
    var ticking = false;

    function frame() {
      ticking = false;
      var y = window.scrollY || window.pageYOffset;
      if (nav) nav.classList.toggle("solid", y > 90);
      if (bar) {
        var h = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.width = (h > 0 ? Math.min(100, (y / h) * 100) : 0) + "%";
      }
      var best = -1, bestTop = -Infinity;
      for (var i = 0; i < targets.length; i++) {
        if (!targets[i]) continue;
        var top = targets[i].getBoundingClientRect().top - 140;
        if (top <= 0 && top > bestTop) { bestTop = top; best = i; }
      }
      links.forEach(function (a, i) { a.classList.toggle("on", i === best); });
    }
    addEventListener("scroll", function () {
      if (!ticking) { ticking = true; requestAnimationFrame(frame); }
    }, { passive: true });
    addEventListener("resize", frame, { passive: true });
    frame();
  })();

  /* ------------------------------------------------ scrollytelling */
  (function scrolly() {
    var steps = [].slice.call(document.querySelectorAll("[data-step]"));
    if (!steps.length) return;
    var onStep = function (i) {
      steps.forEach(function (s, j) {
        s.querySelector(".stepcard").classList.toggle("on", j === i);
      });
      if (window.bcfScrollyStep) window.bcfScrollyStep(i);
    };
    if (reduced || !("IntersectionObserver" in window)) {
      steps.forEach(function (s) { s.querySelector(".stepcard").classList.add("on"); });
      if (window.bcfScrollyStep) window.bcfScrollyStep(steps.length - 1);
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) onStep(steps.indexOf(e.target));
      });
    }, { rootMargin: "-45% 0px -45% 0px", threshold: 0 });
    steps.forEach(function (s) { obs.observe(s); });
    onStep(0);
  })();

  /* ------------------------------------------------- the hero forest */
  (function hero() {
    var canvas = document.getElementById("forestGL");
    var host = document.getElementById("hero");
    if (!canvas || !host || typeof window.initForest !== "function") return;

    var lowPower = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;
    var scale = lowPower ? 0.62 : 0.88;
    var steps = lowPower ? 12 : 16;
    if (innerWidth < 720) { scale = Math.min(scale, 0.72); steps = Math.min(steps, 13); }

    var forest = window.initForest(canvas, { scale: scale, steps: steps });
    if (!forest) return;                       /* poster image stays */

    /* reduced motion still gets the scene, just held still */
    if (reduced) {
      forest.render(6.2);
      canvas.classList.add("on");
      return;
    }

    var t0 = performance.now(), visible = true, raf = 0;
    var walkTarget = 0, parTarget = [0, 0];
    var slowFrames = 0, degraded = 0, lastT = t0;

    var io = new IntersectionObserver(function (e) {
      visible = e[0].isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(loop);
    }, { threshold: 0.01 });
    io.observe(host);

    addEventListener("scroll", function () {
      /* scrolling walks the camera forward down the path */
      walkTarget = (window.scrollY || 0) * 0.0085;
    }, { passive: true });

    if (matchMedia("(hover: hover)").matches) {
      addEventListener("pointermove", function (e) {
        parTarget[0] = (e.clientX / innerWidth) * 2 - 1;
        parTarget[1] = (e.clientY / innerHeight) * 2 - 1;
      }, { passive: true });
    }

    /* The camera drifts slowly and the fog is soft, so 30 fps is
       indistinguishable from 60 here and halves the GPU load. That buys the
       resolution back on a fanless machine instead of spending it on heat. */
    var FRAME_MS = 1000 / 30, acc = 0;

    function loop(now) {
      raf = 0;
      if (!visible) return;
      var dt = now - lastT; lastT = now;
      acc += dt;
      if (acc < FRAME_MS) { raf = requestAnimationFrame(loop); return; }
      acc = Math.min(acc - FRAME_MS, FRAME_MS);

      /* Ignore the first couple of seconds: fonts, charts and layout are all
         competing then, and a slow frame there says nothing about the shader. */
      var settled = (now - t0) > 2200;
      if (settled && dt > FRAME_MS * 1.8 && degraded < 2) {
        if (++slowFrames > 45) {
          degraded++; slowFrames = 0;
          forest.steps = Math.max(11, forest.steps - 3);
          forest.setScale(scale * (degraded === 1 ? 0.78 : 0.60));
        }
      } else if (dt < FRAME_MS * 1.3) { slowFrames = Math.max(0, slowFrames - 1); }

      forest.walk += (walkTarget - forest.walk) * 0.06;
      forest.par[0] += (parTarget[0] - forest.par[0]) * 0.045;
      forest.par[1] += (parTarget[1] - forest.par[1]) * 0.045;
      forest.render((now - t0) / 1000);
      raf = requestAnimationFrame(loop);
    }

    /* one frame before revealing, so the fade never shows an empty buffer */
    forest.render(0);
    requestAnimationFrame(function () {
      canvas.classList.add("on");
      lastT = performance.now();
      raf = requestAnimationFrame(loop);
    });
  })();

  /* ------------------------------------- copy-to-clipboard on <pre> */
  document.querySelectorAll("pre[data-copy]").forEach(function (pre) {
    pre.addEventListener("click", function () {
      var txt = pre.textContent;
      if (navigator.clipboard) navigator.clipboard.writeText(txt);
      var was = pre.dataset.copy;
      pre.dataset.copy = "copied";
      setTimeout(function () { pre.dataset.copy = was; }, 1400);
    });
  });
})();
