/**
 * Performance Monitor Utility for Jetson Orin Nano
 * 
 * Tracks:
 *   - Frames per second (FPS)
 *   - Inference latency
 *   - Buffer fullness
 *   - GPU/CPU utilization estimate
 * 
 * Usage:
 *   const monitor = PerformanceMonitor();
 *   monitor.frameProcessed();     // Call when frame is processed
 *   monitor.inferenceStart();     // Call before inference
 *   monitor.inferenceEnd();       // Call after inference
 *   monitor.log();                // Log metrics to console
 *   monitor.getStats();           // Get current stats object
 */

const PerformanceMonitor = (() => {
  return function() {
    const stats = {
      frameCount: 0,
      startTime: Date.now(),
      inferenceFrames: 0,
      totalInferenceTime: 0,
      inferenceTimings: [],
      lastFrameTime: Date.now(),
      fps: 0,
      avgInferenceMs: 0,
      bufferSize: 0
    };

    let inferenceStart = null;

    return {
      frameProcessed() {
        const now = Date.now();
        const deltaMs = now - stats.lastFrameTime;
        stats.lastFrameTime = now;
        stats.frameCount++;

        // Calculate FPS (simple moving average every 30 frames)
        if (stats.frameCount % 30 === 0) {
          const elapsedSec = (now - stats.startTime) / 1000;
          stats.fps = (stats.frameCount / elapsedSec).toFixed(1);
        }
      },

      inferenceStart() {
        inferenceStart = Date.now();
      },

      inferenceEnd() {
        if (inferenceStart !== null) {
          const duration = Date.now() - inferenceStart;
          stats.totalInferenceTime += duration;
          stats.inferenceFrames++;
          stats.inferenceTimings.push(duration);

          // Keep only last 100 timings for memory efficiency
          if (stats.inferenceTimings.length > 100) {
            stats.inferenceTimings.shift();
          }

          stats.avgInferenceMs = (
            stats.totalInferenceTime / stats.inferenceFrames
          ).toFixed(2);

          inferenceStart = null;
        }
      },

      updateBufferSize(size) {
        stats.bufferSize = size;
      },

      log() {
        const uptime = ((Date.now() - stats.startTime) / 1000).toFixed(1);
        const minInf = Math.min(...stats.inferenceTimings).toFixed(2);
        const maxInf = Math.max(...stats.inferenceTimings).toFixed(2);

        console.group('📊 Performance Monitor');
        console.log(`Uptime: ${uptime}s`);
        console.log(`FPS: ${stats.fps}`);
        console.log(`Total Frames: ${stats.frameCount}`);
        console.log(`Buffer Size: ${stats.bufferSize}`);
        console.log(`Inference Frames: ${stats.inferenceFrames}`);
        console.log(`Avg Inference: ${stats.avgInferenceMs}ms`);
        console.log(`Min/Max Inference: ${minInf}ms / ${maxInf}ms`);

        // Performance assessment
        const fps = parseFloat(stats.fps);
        let assessment;
        if (fps >= 20) {
          assessment = '✅ EXCELLENT - Jetson performing well';
        } else if (fps >= 15) {
          assessment = '✔️ GOOD - Acceptable performance';
        } else if (fps >= 10) {
          assessment = '⚠️ OK - May need optimization';
        } else {
          assessment = '❌ POOR - Reduce resolution or model complexity';
        }
        console.log(`Assessment: ${assessment}`);
        console.groupEnd();
      },

      getStats() {
        return { ...stats };
      },

      reset() {
        stats.frameCount = 0;
        stats.startTime = Date.now();
        stats.inferenceFrames = 0;
        stats.totalInferenceTime = 0;
        stats.inferenceTimings = [];
        stats.fps = 0;
        stats.avgInferenceMs = 0;
      }
    };
  };
})();

/**
 * Usage example:
 * 
 * const monitor = PerformanceMonitor();
 * 
 * // In camera frame callback:
 * holistic.onResults(res => {
 *   monitor.frameProcessed();
 *   monitor.updateBufferSize(buffer.length);
 *   
 *   if (state === STATE_CAPTURE) {
 *     monitor.inferenceStart();
 *     // ... process frame ...
 *     monitor.inferenceEnd();
 *   }
 * });
 * 
 * // Periodically log stats:
 * setInterval(() => {
 *   monitor.log();
 * }, 10000);
 */
