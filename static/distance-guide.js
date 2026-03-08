/**
 * Distance Guide for MediaPipe Holistic
 *
 * Draws a human silhouette overlay on the camera canvas that the user fits into.
 * The silhouette color indicates whether the user is at the optimal distance:
 *   - Green:  correct distance (shoulder ratio 0.18–0.28)
 *   - Yellow: slightly off (0.13–0.18 or 0.28–0.38)
 *   - Red:    too far (<0.13) or too close (>0.38)
 *
 * Camera: IMX477-160 12.3 MP (118° HFOV, 87° VFOV)
 * Optimal distance: ~0.5–0.7 m (1.5–2.3 ft)
 * Shoulder width ratio = |landmark[11].x − landmark[12].x| in normalised coords
 */
const DistanceGuide = (() => {
  // Shoulder-width-to-frame-width ratio thresholds
  const GREEN_MIN  = 0.18;
  const GREEN_MAX  = 0.28;
  const YELLOW_MIN = 0.13;
  const YELLOW_MAX = 0.38;

  const COLOR_GREEN   = '#06D6A0';
  const COLOR_YELLOW  = '#FFD93D';
  const COLOR_RED     = '#FF6B9D';
  const COLOR_NEUTRAL = '#FFFFFF';

  function getStatus(shoulderRatio) {
    if (shoulderRatio === null) {
      return { color: COLOR_NEUTRAL, text: 'Position yourself in the outline', level: 'none', alpha: 0.30 };
    }
    if (shoulderRatio >= GREEN_MIN && shoulderRatio <= GREEN_MAX) {
      return { color: COLOR_GREEN, text: '\u2713 Perfect Distance', level: 'green', alpha: 0.30 };
    }
    if (shoulderRatio < GREEN_MIN && shoulderRatio >= YELLOW_MIN) {
      return { color: COLOR_YELLOW, text: '\u2197 Move a bit closer', level: 'yellow', alpha: 0.50 };
    }
    if (shoulderRatio > GREEN_MAX && shoulderRatio <= YELLOW_MAX) {
      return { color: COLOR_YELLOW, text: '\u2199 Step back a little', level: 'yellow', alpha: 0.50 };
    }
    if (shoulderRatio < YELLOW_MIN) {
      return { color: COLOR_RED, text: '\u26A0 Too far \u2013 move closer', level: 'red', alpha: 0.60 };
    }
    return { color: COLOR_RED, text: '\u26A0 Too close \u2013 step back', level: 'red', alpha: 0.60 };
  }

  function measureShoulders(poseLandmarks) {
    if (!poseLandmarks || poseLandmarks.length <= 12) return null;
    var lsh = poseLandmarks[11];
    var rsh = poseLandmarks[12];
    if (!lsh || !rsh) return null;
    if (!Number.isFinite(lsh.x) || !Number.isFinite(rsh.x)) return null;
    if (!Number.isFinite(lsh.y) || !Number.isFinite(rsh.y)) return null;
    var ratio = Math.abs(lsh.x - rsh.x);
    if (ratio < 0.01) return null;
    return ratio;
  }

  /**
   * Draw the distance guide on a canvas context.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} W  canvas pixel width
   * @param {number} H  canvas pixel height
   * @param {Array|null} poseLandmarks  MediaPipe pose landmarks (normalised 0-1)
   */
  function draw(ctx, W, H, poseLandmarks) {
    var shoulderRatio = measureShoulders(poseLandmarks);
    var status = getStatus(shoulderRatio);

    drawSilhouette(ctx, W, H, status.color, status.alpha);
    drawBadge(ctx, W, H, status);
  }

  function drawSilhouette(ctx, W, H, color, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, W * 0.005);
    ctx.setLineDash([Math.max(6, W * 0.015), Math.max(4, W * 0.01)]);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    var rectW = 0.36 * W;
    var rectH = 0.70 * H;
    var rectX = (W - rectW) / 2;
    var rectY = 0.08 * H;
    var r = Math.min(rectW, rectH) * 0.04;

    ctx.beginPath();
    ctx.moveTo(rectX + r, rectY);
    ctx.lineTo(rectX + rectW - r, rectY);
    ctx.arcTo(rectX + rectW, rectY, rectX + rectW, rectY + r, r);
    ctx.lineTo(rectX + rectW, rectY + rectH - r);
    ctx.arcTo(rectX + rectW, rectY + rectH, rectX + rectW - r, rectY + rectH, r);
    ctx.lineTo(rectX + r, rectY + rectH);
    ctx.arcTo(rectX, rectY + rectH, rectX, rectY + rectH - r, r);
    ctx.lineTo(rectX, rectY + r);
    ctx.arcTo(rectX, rectY, rectX + r, rectY, r);
    ctx.closePath();
    ctx.stroke();

    ctx.restore();
  }

  function drawBadge(ctx, W, H, status) {
    ctx.save();
    var text = status.text;
    var fontSize = Math.max(11, Math.min(14, W * 0.028));
    ctx.font = '700 ' + fontSize + 'px Inter, system-ui, sans-serif';
    var metrics = ctx.measureText(text);
    var textW = metrics.width;
    var padX = fontSize * 0.8;
    var padY = fontSize * 0.45;
    var boxW = textW + padX * 2;
    var boxH = fontSize + padY * 2;
    var boxX = (W - boxW) / 2;
    var boxY = 0.92 * H - boxH;
    var radius = boxH / 2;

    ctx.globalAlpha = status.level === 'green' ? 0.55 : 0.80;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.beginPath();
    ctx.moveTo(boxX + radius, boxY);
    ctx.lineTo(boxX + boxW - radius, boxY);
    ctx.arcTo(boxX + boxW, boxY, boxX + boxW, boxY + radius, radius);
    ctx.arcTo(boxX + boxW, boxY + boxH, boxX + boxW - radius, boxY + boxH, radius);
    ctx.lineTo(boxX + radius, boxY + boxH);
    ctx.arcTo(boxX, boxY + boxH, boxX, boxY + boxH - radius, radius);
    ctx.arcTo(boxX, boxY, boxX + radius, boxY, radius);
    ctx.closePath();
    ctx.fill();

    ctx.fillStyle = status.color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, W / 2, boxY + boxH / 2);

    ctx.restore();
  }

  return { draw: draw, getStatus: getStatus, measureShoulders: measureShoulders };
})();
