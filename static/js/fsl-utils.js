/**
 * FSL Shared Utilities — common functions used by auto.html and activity.html
 * Extracted to eliminate ~300 lines of duplication per template.
 */
const FSL = (() => {
  'use strict';

  /* ========== Hand / Pose helpers ========== */

  function flattenHand(lms) {
    if (!lms) return new Float32Array(63);
    const a = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      const p = lms[i];
      a[i * 3]     = p.x;
      a[i * 3 + 1] = p.y;
      a[i * 3 + 2] = p.z;
    }
    return a;
  }

  function lm(list, idx) {
    if (!list || idx >= list.length) return { x: NaN, y: NaN, z: NaN };
    return list[idx];
  }

  function dist2D(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y;
    return Math.hypot(dx, dy) || 1e-6;
  }

  function normalizeGlobal(hand63, anchors) {
    if (!hand63 || hand63.length !== 63 || !anchors.L_SH || !anchors.R_SH) return new Float32Array(63);
    const out = new Float32Array(63);
    const Cx = (anchors.L_SH.x + anchors.R_SH.x) / 2;
    const Cy = (anchors.L_SH.y + anchors.R_SH.y) / 2;
    const Cz = (anchors.L_SH.z + anchors.R_SH.z) / 2;
    const scale = Math.max(1e-6, dist2D(anchors.L_SH, anchors.R_SH));
    for (let i = 0; i < 63; i += 3) {
      out[i]     = (hand63[i]     - Cx) / scale;
      out[i + 1] = (hand63[i + 1] - Cy) / scale;
      out[i + 2] = (hand63[i + 2] - Cz) / scale;
    }
    return out;
  }

  function clampNormalized(hand63) {
    // Clamp values to [-5, 5] for stability
    const out = new Float32Array(63);
    for (let i = 0; i < 63; i++) out[i] = Math.max(-5, Math.min(5, hand63[i]));
    return out;
  }

  const _ALT_SEL = [0, 4, 8, 12, 16, 20];
  function derivedAltitudeFeatures(L, R, anchors) {
    const out = new Float32Array(60); // 2 hands × 6 landmarks × 5 features
    const brow_y = 0.5 * (anchors.brow_r.y + anchors.brow_l.y);
    let idx = 0;
    for (const H of [L, R]) {
      for (const j of _ALT_SEL) {
        const py = H[j * 3 + 1], pz = H[j * 3 + 2];
        out[idx++] = py - anchors.chin.y;
        out[idx++] = py - anchors.lip_u.y;
        out[idx++] = py - brow_y;
        out[idx++] = py - anchors.forehead.y;
        out[idx++] = pz - anchors.nose.z;
      }
    }
    return out;
  }

  function extractAnchors(pose, face) {
    return {
      L_SH: lm(pose, 11), R_SH: lm(pose, 12),
      nose: lm(face, 1), forehead: lm(face, 10),
      lip_u: lm(face, 13), brow_r: lm(face, 65), brow_l: lm(face, 295),
      chin: lm(face, 152)
    };
  }

  // Shared zero-buffer for missing hands (avoids allocation per frame)
  const _ZERO63 = new Float32Array(63);

  function packFeature(res) {
    const pose = res.poseLandmarks || [], face = res.faceLandmarks || [];
    const anchors = extractAnchors(pose, face);
    const Lh = res.rightHandLandmarks ? flattenHand(res.rightHandLandmarks) : null;
    const Rh = res.leftHandLandmarks  ? flattenHand(res.leftHandLandmarks)  : null;
    if (!Lh && !Rh) return null;
    const L = Lh ? normalizeGlobal(Lh, anchors) : _ZERO63;
    const R = Rh ? normalizeGlobal(Rh, anchors) : _ZERO63;
    const alt = derivedAltitudeFeatures(L, R, anchors);
    // TypedArray.set() — avoids spread-operator intermediate array (188 elements)
    const out = new Float32Array(188);
    out.set(L, 0);       // [0..62]
    out.set(R, 63);      // [63..125]
    out.set(alt, 126);   // [126..185]
    out[186] = Lh ? 1 : 0;
    out[187] = Rh ? 1 : 0;
    return out;
  }

  /* ========== Temporal alignment ========== */

  /**
   * Flatten an array of Float32Arrays into a single plain array for JSON.
   * Avoids flatMap + Array.from overhead (48 intermediate arrays).
   */
  function flattenSequence(frames) {
    const dim = frames[0].length;
    const out = new Float32Array(frames.length * dim);
    for (let i = 0; i < frames.length; i++) {
      out.set(frames[i], i * dim);
    }
    return Array.from(out);           // JSON.stringify needs plain array
  }

  function temporalFixSimple(frames, n) {
    if (frames.length <= n) {
      const pad = Array.from({ length: n - frames.length }, () => frames[frames.length - 1]);
      return frames.concat(pad);
    }
    const step = frames.length / n;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = frames[Math.min(Math.floor(i * step), frames.length - 1)];
    }
    return out;
  }

  function temporalFixMotion(frames, seqLen) {
    if (frames.length <= seqLen) {
      const pad = Array.from({ length: seqLen - frames.length }, () => frames[frames.length - 1]);
      return frames.concat(pad);
    }
    const scores = new Float32Array(frames.length);
    for (let i = 1; i < frames.length; i++) {
      const f0 = frames[i - 1], f1 = frames[i];
      if (!f0 || !f1) continue;
      let motion = 0;
      const len = f0.length - 2;
      for (let j = 0; j < len; j++) motion += Math.abs(f1[j] - f0[j]);
      motion /= len;
      const presence = f1[f1.length - 2] + f1[f1.length - 1];
      scores[i] = 0.7 * motion + 0.3 * presence;
    }
    const W = Math.min(40, frames.length);
    let bestStart = 0, bestSum = -1e9, runSum = 0;
    for (let i = 0; i < W; i++) runSum += scores[i];
    bestSum = runSum;
    for (let i = 1; i <= scores.length - W; i++) {
      runSum += scores[i + W - 1] - scores[i - 1];
      if (runSum > bestSum) { bestSum = runSum; bestStart = i; }
    }
    const selected = frames.slice(bestStart, bestStart + W);
    const step = selected.length / seqLen;
    const aligned = new Array(seqLen);
    for (let i = 0; i < seqLen; i++) {
      aligned[i] = selected[Math.min(Math.floor(i * step), selected.length - 1)];
    }
    return aligned;
  }

  /* ========== Label normalization ========== */

  const LABEL_MAP = {
    "1":"one","2":"two","3":"three","4":"four","5":"five",
    "grandmother":"grandmother","grandma":"grandmother","lola":"grandmother",
    "grandfather":"grandfather","grandpa":"grandfather","lolo":"grandfather",
    "mom":"mother","mother":"mother","nanay":"mother",
    "dad":"father","father":"father","tatay":"father",
    "boy":"boy","girl":"girl","man":"man","woman":"woman",
    "son":"son","daughter":"daughter",
    "black":"black","blue":"blue","green":"green","orange":"orange",
    "pink":"pink","red":"red","white":"white","yellow":"yellow",
    "correct":"correct","dontunderstand":"dontunderstand","no":"no",
    "understand":"understand","wrong":"wrong","yes":"yes"
  };

  function normLabel(s) {
    if (!s) return "";
    const x = String(s).toLowerCase()
      .replace(/^(numbers_|colors_|color_|family_|relationship_|survival_)/, "")
      .replace(/[^a-z0-9]+/g, '');
    return LABEL_MAP[x] || x;
  }

  function inferCategory(label) {
    const raw = String(label).toLowerCase();
    if (raw.startsWith("numbers_")) return "numbers";
    if (raw.startsWith("colors_") || raw.startsWith("color_")) return "colors";
    if (raw.startsWith("family_")) return "family";
    if (raw.startsWith("relationship_")) return "relationship";
    if (raw.startsWith("survival_")) return "survival";
    const l = normLabel(label);
    if (["one","two","three","four","five"].includes(l)) return "numbers";
    if (["daughter","father","grandfather","grandmother","mother","son"].includes(l)) return "family";
    if (["boy","girl","man","woman"].includes(l)) return "relationship";
    if (["correct","dontunderstand","no","understand","wrong","yes"].includes(l)) return "survival";
    return "colors";
  }

  /* ========== Video helpers ========== */

  async function findDemoUrl(cat, key) {
    const STATIC_BASE = (window.location.origin + '/static/video').replace(/\/+$/, '');
    const candidates = [];
    for (let i = 1; i <= 10; i++) {
      candidates.push(`${key}_${String(i).padStart(2, '0')}.mp4`);
    }
    candidates.push(`${key}.mp4`);
    for (const n of candidates) {
      const url = `${STATIC_BASE}/${cat}/${n}`;
      try {
        const r = await fetch(url, { method: 'HEAD', cache: 'no-store' });
        if (r.ok) return url;
      } catch (_) {}
    }
    return null;
  }

  /* ========== Drawing helpers ========== */

  const HAND_CONN = [
    [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
  ];

  function drawHandBox(landmarks, ctx, canvasW, canvasH, color) {
    if (!landmarks || !landmarks.length) return;
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of landmarks) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.strokeRect(minX * canvasW, minY * canvasH, (maxX - minX) * canvasW, (maxY - minY) * canvasH);
    ctx.fillStyle = color;
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * canvasW, p.y * canvasH, 6, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function drawGhostHand(norm63, anchors, ctx, canvasW, canvasH, color, scaleMultiplier, yOffset) {
    if (!norm63) return;
    const baseScale = Math.max(1e-6, dist2D(anchors.L_SH, anchors.R_SH));
    const scale = baseScale * scaleMultiplier;
    const Cx = (anchors.L_SH.x + anchors.R_SH.x) / 2;
    const Cy = (anchors.L_SH.y + anchors.R_SH.y) / 2 + yOffset;
    const pts = new Array(21);
    for (let i = 0; i < 63; i += 3) {
      pts[i / 3] = { x: norm63[i] * scale + Cx, y: norm63[i + 1] * scale + Cy };
    }
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 6;
    for (const [a, b] of HAND_CONN) {
      const p = pts[a], q = pts[b];
      ctx.beginPath();
      ctx.moveTo(p.x * canvasW, p.y * canvasH);
      ctx.lineTo(q.x * canvasW, q.y * canvasH);
      ctx.stroke();
    }
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * canvasW, p.y * canvasH, 8, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.restore();
  }

  function anchorsValid(a) {
    if (!a || !a.L_SH || !a.R_SH) return false;
    return Number.isFinite(a.L_SH.x) && Number.isFinite(a.R_SH.x) &&
           Number.isFinite(a.L_SH.y) && Number.isFinite(a.R_SH.y) &&
           dist2D(a.L_SH, a.R_SH) > 0.02;
  }

  /* ========== FPS counter ========== */

  function createFPSTracker(el) {
    let lastTime = Date.now(), frameCount = 0;
    return function update() {
      frameCount++;
      const now = Date.now();
      const delta = now - lastTime;
      if (delta >= 500) {
        el.textContent = 'FPS: ' + Math.round((frameCount * 1000) / delta);
        frameCount = 0;
        lastTime = now;
      }
    };
  }

  /* ========== Ghost Loader ========== */

  async function loadGhostFrames(label, holisticOptions) {
    const cat = inferCategory(label);
    const key = normLabel(label);
    const url = await findDemoUrl(cat, key);
    if (!url) return null;

    const v = document.createElement('video');
    v.src = url; v.muted = true; v.playsInline = true; v.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => { v.onloadedmetadata = resolve; v.onerror = () => reject(new Error('Video error')); });

    const ghostHolistic = new Holistic({ locateFile: (f) => `/static/mediapipe/holistic/${f}` });
    ghostHolistic.setOptions(holisticOptions);

    const frames = [];
    ghostHolistic.onResults(res => {
      const pose = res.poseLandmarks || [], face = res.faceLandmarks || [];
      const anchors = extractAnchors(pose, face);
      const Lh = res.rightHandLandmarks ? flattenHand(res.rightHandLandmarks) : null;
      const Rh = res.leftHandLandmarks  ? flattenHand(res.leftHandLandmarks)  : null;
      if (!Lh && !Rh) return;
      const L = Lh ? normalizeGlobal(Lh, anchors) : null;
      const R = Rh ? normalizeGlobal(Rh, anchors) : null;
      frames.push({ L, R });
    });

    const total = Math.min(24, Math.max(12, Math.floor(v.duration * 6)));
    for (let i = 0; i < total; i++) {
      v.currentTime = (v.duration * i) / total;
      await new Promise(r => { v.onseeked = r; });
      await ghostHolistic.send({ image: v });
    }

    return frames.length > 0 ? frames : null;
  }

  function interpolateGhostFrame(frames, ghostIdx) {
    const total = frames.length;
    const idxA = Math.floor(ghostIdx) % total;
    const idxB = Math.floor(ghostIdx + 1) % total;
    const t = ghostIdx % 1;
    const frameA = frames[idxA], frameB = frames[idxB];
    let L = null, R = null;
    if (frameA.L && frameB.L) {
      L = new Float32Array(63);
      for (let i = 0; i < 63; i++) L[i] = (1 - t) * frameA.L[i] + t * frameB.L[i];
    }
    if (frameA.R && frameB.R) {
      R = new Float32Array(63);
      for (let i = 0; i < 63; i++) R[i] = (1 - t) * frameA.R[i] + t * frameB.R[i];
    }
    return { L, R };
  }

  /* ========== Public API ========== */

  return {
    flattenHand, lm, dist2D, normalizeGlobal, clampNormalized,
    derivedAltitudeFeatures, extractAnchors, packFeature,
    flattenSequence, temporalFixSimple, temporalFixMotion,
    normLabel, inferCategory,
    findDemoUrl,
    HAND_CONN, drawHandBox, drawGhostHand, anchorsValid,
    createFPSTracker,
    loadGhostFrames, interpolateGhostFrame,
  };
})();
