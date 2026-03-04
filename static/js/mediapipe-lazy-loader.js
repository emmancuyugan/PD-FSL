/**
 * Lazy Load Utility for Mediapipe Assets
 * 
 * Usage:
 *   - Call loadMediapipeScripts() when user navigates to detection page
 *   - All 3 Mediapipe scripts loaded in parallel async
 * 
 * Benefits:
 *   - Faster initial page load
 *   - Scripts only loaded when needed
 *   - Parallel loading reduces total load time
 */

const MediapipeLazyLoader = (() => {
  let scriptsLoaded = false;
  let loadPromise = null;

  const SCRIPT_URLs = [
    '/static/mediapipe/camera_utils/camera_utils.js',
    '/static/mediapipe/drawing_utils/drawing_utils.js',
    '/static/mediapipe/holistic/holistic.js'
  ];

  /**
   * Load all Mediapipe scripts in parallel
   * @returns {Promise} Resolves when all scripts are loaded
   */
  function loadMediapipeScripts() {
    if (scriptsLoaded) {
      return Promise.resolve();
    }

    if (loadPromise) {
      return loadPromise;
    }

    loadPromise = Promise.all(
      SCRIPT_URLs.map(url =>
        new Promise((resolve, reject) => {
          // Check if already loaded
          if (document.querySelector(`script[src="${url}"]`)) {
            resolve();
            return;
          }

          const script = document.createElement('script');
          script.src = url;
          script.async = true;
          script.onload = resolve;
          script.onerror = () => reject(new Error(`Failed to load ${url}`));
          document.head.appendChild(script);
        })
      )
    ).then(() => {
      scriptsLoaded = true;
    });

    return loadPromise;
  }

  /**
   * Preload Mediapipe scripts in background (when page is idle)
   * Uses requestIdleCallback if available, falls back to setTimeout
   */
  function preloadMediapipeScripts() {
    if (scriptsLoaded || loadPromise) {
      return;
    }

    const preloadFunc = () => {
      loadMediapipeScripts().catch(err => {
        console.warn('Mediapipe preload failed:', err);
        // Preload failure is non-critical, don't throw
      });
    };

    if ('requestIdleCallback' in window) {
      requestIdleCallback(preloadFunc, { timeout: 2000 });
    } else {
      // Fallback: load after 2 seconds
      setTimeout(preloadFunc, 2000);
    }
  }

  return {
    loadMediapipeScripts,
    preloadMediapipeScripts,
    isLoaded: () => scriptsLoaded
  };
})();

/**
 * Optional: Automatically preload when page is ready
 * Uncomment to enable:
 */
// document.addEventListener('DOMContentLoaded', () => {
//   MediapipeLazyLoader.preloadMediapipeScripts();
// });
