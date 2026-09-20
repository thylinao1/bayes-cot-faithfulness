/* charts.js — the page's live figures.

   Every number here is transcribed from a project artifact; the sweep
   arrays are the same precomputed vectors the previous site shipped,
   which came out of the Python package (notebook 04 config).

   Palette note: orange = decorative/NDE, cyan = faithful/NIE, green =
   total effect are fixed by notebooks/figstyle.py and the published
   PNGs depend on them, so they are not retuned here. They clear CVD
   separation comfortably (worst adjacent pair dE 15.9); their one
   weakness is that all three sit bright, so green never carries a
   large fill, and every series is direct-labelled as well as keyed. */
(function () {
  "use strict";

  var C = {
    nde: "#ff8c42", nie: "#4dd0e1", te: "#a5e887",
    ink: "#ffffff", ink2: "#9db0a4", ink3: "#6d8177",
    grid: "#1c2a24", rule: "#243530", surface: "#0a110f",
    gold: "#cf9a2f", clay: "#e0724a"
  };
  var MONO = '11px "IBM Plex Mono", ui-monospace, Menlo, monospace';
  var MONO_S = '10px "IBM Plex Mono", ui-monospace, Menlo, monospace';
  var MONO_L = '12.5px "IBM Plex Mono", ui-monospace, Menlo, monospace';

  var reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ease = function (t) { return 1 - Math.pow(1 - t, 3); };
  var fmt = function (x, d) {
    d = d === undefined ? 2 : d;
    return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
  };

  function rrect(g, x, y, w, h, r) {
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    g.beginPath();
    g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r);
    g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r);
    g.arcTo(x, y, x + w, y, r); g.closePath();
  }

  /* ---------------------------------------------------------------
     chart shell: DPR sizing, enter animation, hover plumbing
     --------------------------------------------------------------- */
  function mount(canvas, opts) {
    var g = canvas.getContext("2d");
    var W = 0, H = 0, p = reduced ? 1 : 0, started = reduced, hover = null;
    var api = { hit: [], data: opts.data };

    function size() {
      var dpr = Math.min(devicePixelRatio || 1, 2);
      var w = canvas.clientWidth, h = opts.height;
      canvas.style.height = h + "px";
      if (w === W && canvas.height === Math.round(h * dpr)) return;
      W = w; H = h;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function paint() {
      size();
      g.clearRect(0, 0, W, H);
      api.hit = [];
      opts.draw(g, W, H, p, api, hover);
      if (hover && opts.tip) drawTip(g, W, H, hover);
    }

    function drawTip(g, W, H, h) {
      if (!h.lines || !h.lines.length) return;
      g.font = MONO;
      var pad = 9, lh = 15;
      var w = 0;
      h.lines.forEach(function (l) { w = Math.max(w, g.measureText(l).width); });
      w += pad * 2;
      var th = h.lines.length * lh + pad * 2 - 3;
      var x = Math.min(Math.max(h.x + 14, 4), W - w - 4);
      var y = Math.min(Math.max(h.y - th - 10, 4), H - th - 4);
      g.fillStyle = "rgba(8,14,12,.96)";
      g.strokeStyle = C.rule; g.lineWidth = 1;
      rrect(g, x, y, w, th, 5); g.fill(); g.stroke();
      h.lines.forEach(function (l, i) {
        g.fillStyle = i === 0 ? C.ink : C.ink2;
        g.fillText(l, x + pad, y + pad + 11 + i * lh);
      });
    }

    api.repaint = paint;

    if (!reduced && "IntersectionObserver" in window) {
      new IntersectionObserver(function (e, o) {
        if (!e[0].isIntersecting || started) return;
        started = true; o.disconnect();
        var t0 = performance.now();
        (function step(now) {
          p = Math.min(1, (now - t0) / 950);
          paint();
          if (p < 1) requestAnimationFrame(step);
        })(t0);
      }, { threshold: 0.25 }).observe(canvas);
    }

    if (opts.tip) {
      canvas.addEventListener("pointermove", function (e) {
        var r = canvas.getBoundingClientRect();
        var mx = e.clientX - r.left, my = e.clientY - r.top;
        var found = null;
        for (var i = 0; i < api.hit.length; i++) {
          var z = api.hit[i];
          if (mx >= z.x && mx <= z.x + z.w && my >= z.y && my <= z.y + z.h) { found = z; break; }
        }
        var changed = (found && (!hover || hover.key !== found.key)) || (!found && hover);
        if (found) hover = { x: mx, y: my, key: found.key, lines: found.lines, idx: found.idx };
        else hover = null;
        if (changed || found) paint();
      });
      canvas.addEventListener("pointerleave", function () {
        if (hover) { hover = null; paint(); }
      });
    }

    addEventListener("resize", function () { W = 0; paint(); }, { passive: true });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(paint);
    paint();
    return api;
  }

  /* =============================================================
     1. the rho dial: what the unverifiable assumption costs
     ============================================================= */
  var SWEEP = {
    rho: [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
    nde: [-0.3686,-0.3238,-0.2810,-0.2404,-0.2006,-0.1612,-0.1195,-0.0783,-0.0328,0.0141,0.0663,0.1242,0.1939,0.2721,0.3698],
    nie: [0.6733,0.6294,0.5868,0.5460,0.5068,0.4679,0.4273,0.3862,0.3408,0.2942,0.2417,0.1843,0.1151,0.0335,-0.0636],
    te:  [0.3046,0.3056,0.3058,0.3056,0.3062,0.3067,0.3078,0.3079,0.3080,0.3083,0.3081,0.3085,0.3090,0.3056,0.3062],
    trueNde: 0.0909, trueNie: 0.2074, rhoStar: 0.7364, rhoTrue: 0.5
  };
  function interp(ys, r) {
    var rs = SWEEP.rho;
    if (r <= rs[0]) return ys[0];
    if (r >= rs[rs.length - 1]) return ys[ys.length - 1];
    var i = 0; while (i < rs.length - 1 && rs[i + 1] < r) i++;
    var t = (r - rs[i]) / (rs[i + 1] - rs[i]);
    return ys[i] + t * (ys[i + 1] - ys[i]);
  }

  function rhoDial() {
    var cv = document.getElementById("cRho");
    if (!cv) return;
    var slider = document.getElementById("rhoSlider");
    var readout = document.getElementById("rhoRead");
    var verdict = document.getElementById("rhoVerdict");
    var rho = 0;

    var ch = mount(cv, {
      height: 400, tip: true,
      draw: function (g, W, H, p, api) {
        var narrow = W < 580;
        var L = narrow ? 42 : 54, R = narrow ? 18 : 118, T = 26, B = 44;
        var iw = W - L - R, ih = H - T - B;
        var x0 = -0.66, x1 = 0.86, y0 = -0.42, y1 = 0.72;
        var sx = function (v) { return L + (v - x0) / (x1 - x0) * iw; };
        var sy = function (v) { return T + (1 - (v - y0) / (y1 - y0)) * ih; };

        g.font = MONO_S; g.textBaseline = "middle";
        [-0.4,-0.2,0,0.2,0.4,0.6].forEach(function (t) {
          var y = sy(t);
          g.strokeStyle = t === 0 ? C.rule : C.grid;
          g.lineWidth = 1; g.beginPath(); g.moveTo(L, y); g.lineTo(L + iw, y); g.stroke();
          g.fillStyle = C.ink3; g.textAlign = "right";
          g.fillText(fmt(t, 1), L - 10, y);
        });
        g.textAlign = "center"; g.textBaseline = "top";
        (narrow ? [-0.6,-0.2,0.2,0.6] : [-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8]).forEach(function (t) {
          g.fillStyle = C.ink3; g.fillText(t.toFixed(1), sx(t), T + ih + 11);
        });
        g.fillStyle = C.ink3; g.font = MONO_S;
        g.fillText("assumed residual correlation  rho", L + iw / 2, T + ih + 28);

        /* the zone past rho*, where the pre-registered verdict fails */
        var xs = sx(SWEEP.rhoStar);
        if (xs < L + iw) {
          g.fillStyle = "rgba(224,114,74,.10)";
          g.fillRect(xs, T, L + iw - xs, ih);
          g.strokeStyle = C.clay; g.lineWidth = 1; g.setLineDash([3, 3]);
          g.beginPath(); g.moveTo(xs, T); g.lineTo(xs, T + ih); g.stroke();
          g.setLineDash([]);
          /* when there is no right margin, hang the label inside the plot */
          var lx = narrow ? xs - 7 : xs + 7;
          g.save(); g.translate(lx, T + 9);
          g.textAlign = narrow ? "right" : "left"; g.textBaseline = "top";
          g.fillStyle = C.clay; g.font = MONO_S;
          g.fillText("rho* = " + SWEEP.rhoStar.toFixed(3), 0, 0);
          g.fillText("verdict flips", 0, 13);
          g.restore();
        }

        var n = SWEEP.rho.length;
        var upto = Math.max(2, Math.ceil(n * p));
        function path(ys, col, lab) {
          g.strokeStyle = col; g.lineWidth = 2;
          g.lineJoin = "round"; g.lineCap = "round";
          g.beginPath();
          for (var i = 0; i < upto; i++) {
            var X = sx(SWEEP.rho[i]), Y = sy(ys[i]);
            i ? g.lineTo(X, Y) : g.moveTo(X, Y);
          }
          g.stroke();
          if (p > 0.72 && !narrow) {
            g.globalAlpha = Math.min(1, (p - 0.72) / 0.28);
            g.fillStyle = col; g.font = MONO; g.textAlign = "left"; g.textBaseline = "middle";
            g.fillText(lab, sx(SWEEP.rho[n - 1]) + 9, sy(ys[n - 1]));
            g.globalAlpha = 1;
          }
        }
        path(SWEEP.te, C.te, "total");
        path(SWEEP.nde, C.nde, "decorative");
        path(SWEEP.nie, C.nie, "faithful");

        /* truth markers: what the world actually contained */
        if (p > 0.85 && !narrow) {
          g.globalAlpha = (p - 0.85) / 0.15;
          [[SWEEP.trueNie, C.nie], [SWEEP.trueNde, C.nde]].forEach(function (d) {
            var X = sx(SWEEP.rhoTrue), Y = sy(d[0]);
            g.strokeStyle = d[1]; g.lineWidth = 1.5;
            g.beginPath(); g.arc(X, Y, 5.5, 0, 6.284); g.stroke();
            g.fillStyle = C.surface; g.fill(); g.stroke();
          });
          g.fillStyle = C.ink3; g.font = MONO_S; g.textAlign = "center"; g.textBaseline = "bottom";
          g.fillText("truth, at the real rho = 0.5", sx(SWEEP.rhoTrue) - 26, sy(SWEEP.trueNie) - 14);
          g.globalAlpha = 1;
        }

        /* the live dial */
        var X = sx(rho);
        g.strokeStyle = "rgba(207,154,47,.85)"; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(X, T); g.lineTo(X, T + ih); g.stroke();
        [[SWEEP.nie, C.nie], [SWEEP.nde, C.nde], [SWEEP.te, C.te]].forEach(function (d) {
          var Y = sy(interp(d[0], rho));
          g.fillStyle = d[1]; g.strokeStyle = C.surface; g.lineWidth = 2;
          g.beginPath(); g.arc(X, Y, 5, 0, 6.284); g.fill(); g.stroke();
        });
        /* one hover band per sweep step, so the tooltip reads real values */
        var bw = iw / (n - 1);
        for (var k = 0; k < n; k++) {
          api.hit.push({
            x: sx(SWEEP.rho[k]) - bw / 2, y: T, w: bw, h: ih,
            key: "r" + k, idx: k,
            lines: ["rho = " + SWEEP.rho[k].toFixed(1),
                    "faithful    " + fmt(SWEEP.nie[k], 4),
                    "decorative  " + fmt(SWEEP.nde[k], 4),
                    "total       " + fmt(SWEEP.te[k], 4)]
          });
        }
      },
      data: SWEEP
    });

    function refresh() {
      if (readout) {
        readout.innerHTML =
          '<span style="color:' + C.nie + '">' + fmt(interp(SWEEP.nie, rho)) + "</span> / " +
          '<span style="color:' + C.nde + '">' + fmt(interp(SWEEP.nde, rho)) + "</span>";
      }
      if (verdict) {
        var live = interp(SWEEP.nie, rho) > 0;
        verdict.textContent = live ? "faithful path still positive" : "faithful path has crossed zero";
        verdict.className = "chip " + (live ? "pass" : "fail");
      }
      ch.repaint();
    }
    if (slider) {
      slider.addEventListener("input", function () {
        rho = parseFloat(slider.value) / 100;
        var lbl = document.getElementById("rhoVal");
        if (lbl) lbl.textContent = (rho >= 0 ? "+" : "−") + Math.abs(rho).toFixed(2);
        refresh();
      });
    }
    refresh();
  }

  /* =============================================================
     2. the anchor and the fit disagree, on every model
     ============================================================= */
  function anchorFit() {
    var cv = document.getElementById("cAnchor");
    if (!cv) return;
    var rows = [
      { m: "qwen3-8b",              anchor: 0.146, fit: 0.0250, lo: -0.0342, hi: 0.1434 },
      { m: "gemma-2-9b-it",         anchor: 0.169, fit: 0.0523, lo: -0.0370, hi: 0.2206 },
      { m: "llama-3.1-8b-instruct", anchor: 0.218, fit: 0.0161, lo: -0.0353, hi: 0.1160 }
    ];
    mount(cv, {
      height: 296, tip: true,
      draw: function (g, W, H, p, api) {
        var narrow = W < 580;
        var L = narrow ? 96 : Math.min(168, W * 0.34), R = narrow ? 14 : 26,
            T = narrow ? 60 : 46, B = 52;
        var iw = W - L - R, ih = H - T - B;
        var x0 = -0.06, x1 = 0.25;
        var sx = function (v) { return L + (v - x0) / (x1 - x0) * iw; };
        var rowH = ih / rows.length;

        g.font = MONO_S; g.textAlign = "center"; g.textBaseline = "top";
        (narrow ? [0, 0.10, 0.20] : [-0.05, 0, 0.05, 0.10, 0.15, 0.20]).forEach(function (t) {
          var x = sx(t);
          g.strokeStyle = t === 0 ? C.rule : C.grid; g.lineWidth = 1;
          g.beginPath(); g.moveTo(x, T - 10); g.lineTo(x, T + ih); g.stroke();
          g.fillStyle = C.ink3; g.fillText(t.toFixed(2), x, T + ih + 10);
        });
        g.fillStyle = C.ink3;
        g.fillText("natural indirect effect  (the faithful path)", L + iw / 2, T + ih + 27);

        rows.forEach(function (r, i) {
          var y = T + rowH * i + rowH / 2;
          var xa = sx(r.anchor), xf = sx(r.fit);
          var q = Math.min(1, Math.max(0, (p - i * 0.12) / 0.62));
          if (q <= 0) return;

          g.font = narrow ? MONO_S : MONO; g.textAlign = "right"; g.textBaseline = "middle";
          g.fillStyle = C.ink2;
          g.fillText(narrow ? r.m.split("-")[0] : r.m, L - 12, y);

          /* the fitted interval, which covers zero on every row */
          g.globalAlpha = q;
          g.strokeStyle = "rgba(77,208,225,.30)"; g.lineWidth = 5; g.lineCap = "round";
          g.beginPath(); g.moveTo(sx(r.lo), y); g.lineTo(sx(r.hi), y); g.stroke();

          /* the gap itself is the subject, so it is drawn as a connector */
          g.strokeStyle = "rgba(224,114,74,.92)"; g.lineWidth = 2;
          g.setLineDash([5, 4]);
          g.beginPath(); g.moveTo(xf, y); g.lineTo(xf + (xa - xf) * q, y); g.stroke();
          g.setLineDash([]);

          g.fillStyle = C.nie; g.strokeStyle = C.surface; g.lineWidth = 2;
          g.beginPath(); g.arc(xf, y, 6.5, 0, 6.284); g.fill(); g.stroke();
          g.fillStyle = C.gold;
          g.beginPath(); g.arc(xf + (xa - xf) * q, y, 6.5, 0, 6.284); g.fill(); g.stroke();
          g.globalAlpha = 1;

          api.hit.push({
            x: L, y: y - rowH / 2, w: iw, h: rowH, key: r.m, idx: i,
            lines: [r.m,
                    "randomized anchor  " + r.anchor.toFixed(3),
                    "fitted NIE         " + fmt(r.fit, 4),
                    "95% CrI  [" + fmt(r.lo, 4) + ", " + fmt(r.hi, 4) + "]"]
          });
        });

        if (p > 0.8) {
          g.globalAlpha = (p - 0.8) / 0.2;
          g.font = MONO; g.textBaseline = "middle";
          g.textAlign = "left"; g.font = narrow ? MONO_S : MONO;
          g.fillStyle = C.gold;
          g.fillText("● randomized anchor", narrow ? 4 : L, T - 26);
          g.fillStyle = C.nie;
          g.fillText("● mediation fit, 95% CrI",
                     narrow ? 4 : L + Math.min(200, iw * 0.44), narrow ? T - 42 : T - 26);
          g.globalAlpha = 1;
        }
      }
    });
  }

  /* =============================================================
     3. batching breaks greedy decoding
     ============================================================= */
  function batching() {
    var cv = document.getElementById("cBatch");
    if (!cv) return;
    var conc = [1, 8, 32, 64];
    var off = [30, 11, 13, 13];
    var on  = [30, 30, 30, 30];
    mount(cv, {
      height: 300, tip: true,
      draw: function (g, W, H, p, api) {
        var narrow = W < 580;
        var L = narrow ? 44 : 74, R = narrow ? 12 : 22,
            T = narrow ? 62 : 48, B = 52;
        var iw = W - L - R, ih = H - T - B;
        var sy = function (v) { return T + (1 - v / 30) * ih; };
        var group = iw / conc.length;
        var bw = Math.min(38, (group - 14) / 2);

        g.font = MONO_S; g.textAlign = "right"; g.textBaseline = "middle";
        [0, 10, 20, 30].forEach(function (t) {
          var y = sy(t);
          g.strokeStyle = t === 0 ? C.rule : C.grid; g.lineWidth = 1;
          g.beginPath(); g.moveTo(L, y); g.lineTo(L + iw, y); g.stroke();
          g.fillStyle = C.ink3; g.fillText(String(t), L - 10, y);
        });
        g.save(); g.translate(narrow ? 11 : 16, T + ih / 2); g.rotate(-Math.PI / 2);
        g.textAlign = "center"; g.textBaseline = "top"; g.fillStyle = C.ink3;
        g.fillText("identical completions, out of 30", 0, 0); g.restore();

        conc.forEach(function (c, i) {
          var cx = L + group * i + group / 2;
          [[off[i], C.nde, -1, "default"], [on[i], C.te, 1, "batch invariant"]].forEach(function (d) {
            var v = d[0] * p, col = d[1];
            var x = cx + (d[2] < 0 ? -bw - 1 : 1);
            var y = sy(v), h = sy(0) - y;
            g.globalAlpha = d[2] < 0 ? 1 : 0.88;
            g.fillStyle = col;
            if (h > 0.5) { rrect(g, x, y, bw, h, 4); g.fill(); }
            g.globalAlpha = 1;
            if (p > 0.7) {
              g.globalAlpha = (p - 0.7) / 0.3;
              g.font = MONO; g.textAlign = "center"; g.textBaseline = "bottom";
              g.fillStyle = col;
              g.fillText(String(d[0]), x + bw / 2, y - 5);
              g.globalAlpha = 1;
            }
            api.hit.push({
              x: x, y: T, w: bw, h: ih, key: d[3] + c, idx: i,
              lines: [d[3] + ", " + c + " in flight",
                      d[0] + " of 30 identical",
                      "max logprob shift " +
                        (d[3] === "default" ? [0, 0.625, 0.875, 0.875][i].toFixed(3) : "0.000") + " nats"]
            });
          });
          g.font = MONO_S; g.textAlign = "center"; g.textBaseline = "top";
          g.fillStyle = C.ink3; g.fillText(String(c), cx, T + ih + 11);
        });
        g.fillStyle = C.ink3; g.font = MONO_S; g.textAlign = "center";
        g.fillText("requests in flight", L + iw / 2, T + ih + 28);

        if (p > 0.8) {
          g.globalAlpha = (p - 0.8) / 0.2;
          g.font = MONO; g.textBaseline = "middle"; g.textAlign = "left";
          g.font = narrow ? MONO_S : MONO;
          g.fillStyle = C.nde; g.fillText("■ default serving", narrow ? 4 : L, T - 26);
          g.fillStyle = C.te;
          g.fillText("■ batch invariant",
                     narrow ? 4 : L + Math.min(180, iw * 0.42), narrow ? T - 42 : T - 26);
          g.globalAlpha = 1;
        }
      }
    });
  }

  /* =============================================================
     4. rho* cannot rank: the largest one sits on zero mediation
     ============================================================= */
  function rhoStarRank() {
    var cv = document.getElementById("cRank");
    if (!cv) return;
    var rows = [
      { w: "shared cause",    rs: 0.7069, truth: "true mediation 0.000", zero: true },
      { w: "rationalization", rs: 0.7063, truth: "fully mediated, 0.283", zero: false },
      { w: "third world",     rs: 0.8428, truth: "true mediation 0.000", zero: true }
    ];
    mount(cv, {
      height: 268, tip: true,
      draw: function (g, W, H, p, api) {
        var narrow = W < 580;
        var L = narrow ? 86 : Math.min(150, W * 0.32), R = narrow ? 62 : 150,
            T = 40, B = 46;
        var iw = W - L - R, ih = H - T - B;
        var sx = function (v) { return L + (v / 1.0) * iw; };
        var rowH = ih / rows.length, bh = Math.min(30, rowH - 18);

        g.font = MONO_S; g.textAlign = "center"; g.textBaseline = "top";
        (narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).forEach(function (t) {
          var x = sx(t);
          g.strokeStyle = t === 0 ? C.rule : C.grid; g.lineWidth = 1;
          g.beginPath(); g.moveTo(x, T - 8); g.lineTo(x, T + ih); g.stroke();
          g.fillStyle = C.ink3; g.fillText(t.toFixed(2), x, T + ih + 10);
        });
        g.fillStyle = C.ink3;
        g.fillText("rho*  (breakdown frontier)", L + iw / 2, T + ih + 27);

        rows.forEach(function (r, i) {
          var y = T + rowH * i + (rowH - bh) / 2;
          var q = Math.min(1, Math.max(0, (p - i * 0.1) / 0.7));
          var col = r.zero ? C.nde : C.te;
          g.font = narrow ? MONO_S : MONO; g.textAlign = "right"; g.textBaseline = "middle";
          g.fillStyle = C.ink2;
          g.fillText(narrow ? r.w.split(" ")[0] : r.w, L - 12, y + bh / 2);

          g.globalAlpha = 0.9 * q; g.fillStyle = col;
          var bwid = (sx(r.rs) - L) * q;
          if (bwid > 1) { rrect(g, L, y, bwid, bh, 4); g.fill(); }
          g.globalAlpha = 1;

          if (p > 0.62) {
            g.globalAlpha = (p - 0.62) / 0.38;
            g.font = MONO; g.textAlign = "left"; g.textBaseline = "middle";
            g.fillStyle = C.ink; g.fillText(r.rs.toFixed(4), sx(r.rs) + 10, y + bh / 2);
            if (!narrow) {
              g.font = MONO_S; g.fillStyle = r.zero ? C.nde : C.ink3;
              g.fillText(r.truth, sx(r.rs) + 10, y + bh / 2 + 14);
            }
            g.globalAlpha = 1;
          }
          api.hit.push({
            x: L, y: y - 6, w: iw, h: bh + 12, key: r.w, idx: i,
            lines: [r.w, "rho* " + r.rs.toFixed(4), r.truth]
          });
        });
      }
    });
  }

  function boot() { rhoDial(); anchorFit(); batching(); rhoStarRank(); }
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", boot);
  else boot();
})();
