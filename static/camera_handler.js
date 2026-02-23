/**
 * Unified Camera Handler for PD-FSL
 * 
 * This module handles camera capture and Mediapipe processing.
 * If C++ optimized backend is available, it uses that (faster, lower latency).
 * Otherwise, it falls back to browser-based JavaScript Mediapipe processing.
 * 
 * Features:
 * - Automatic fallback from C++ to browser Mediapipe
 * - Frame capture and processing with landmarks
 * - Real-time visualization options
 * - Compatible with existing UI
 */

class CameraHandler {
  constructor(options = {}) {
    this.useNative = options.useNative !== false;  // Try C++ by default
    this.usingCPP = false;
    this.usingBrowser = false;
    
    this.videoElement = options.videoElement || null;
    this.canvasElement = options.canvasElement || null;
    this.onFrame = options.onFrame || null;
    
    this.isRunning = false;
    this.mediaStream = null;
    this.frameBuffer = [];
    this.maxBufferSize = options.maxBufferSize || 30;
    
    // Browser fallback state
    this.holistic = null;
    this.camera = null;
    this.cameraBrowser = null;
  }
  
  /**
   * Initialize camera handler
   * Attempts C++ backend first, falls back to browser
   */
  async init() {
    console.log("[CameraHandler] Initializing...");
    
    if (this.useNative) {
      const cppResult = await this.initCPP();
      if (cppResult) {
        this.usingCPP = true;
        console.log("[CameraHandler] Using C++ optimized backend ✅");
        return true;
      }
    }
    
    // Fallback to browser
    console.log("[CameraHandler] Falling back to browser Mediapipe");
    const browserResult = await this.initBrowser();
    if (browserResult) {
      this.usingBrowser = true;
      console.log("[CameraHandler] Using browser Mediapipe ✅");
      return true;
    }
    
    console.error("[CameraHandler] Failed to initialize both backends");
    return false;
  }
  
  /**
   * Initialize C++ backend
   */
  async initCPP() {
    try {
      // Check if backend is available
      const response = await fetch("/ping");
      if (!response.ok) return false;
      
      // Initialize camera
      const initResp = await fetch("/camera/init", { method: "POST" });
      if (!initResp.ok) return false;
      
      const data = await initResp.json();
      if (data.status === "fallback") {
        console.log("[Camera] Backend indicates fallback needed");
        return false;
      }
      
      return true;
    } catch (e) {
      console.warn("[Camera] C++ backend init failed:", e);
      return false;
    }
  }
  
  /**
   * Initialize browser-based Mediapipe
   */
  async initBrowser() {
    try {
      // Load Mediapipe if not already loaded
      if (typeof Holistic === 'undefined') {
        console.log("[Camera] Loading Mediapipe Holistic...");
        
        // Check if script already exists
        const script = document.querySelector('script[src*="holistic.js"]');
        if (!script) {
          const newScript = document.createElement('script');
          newScript.src = '/static/mediapipe/holistic/holistic.js';
          document.head.appendChild(newScript);
          
          // Wait for load
          await new Promise((resolve, reject) => {
            newScript.onload = resolve;
            newScript.onerror = reject;
            setTimeout(reject, 5000);  // 5s timeout
          });
        }
      }
      
      if (typeof Holistic === 'undefined') {
        console.error("[Camera] Mediapipe Holistic not loaded");
        return false;
      }
      
      // Initialize Holistic
      this.holistic = new Holistic({
        locateFile: (f) => `/static/mediapipe/holistic/${f}`
      });
      
      this.holistic.setOptions({
        selfieMode: false,
        modelComplexity: 1,
        smoothLandmarks: true,
        refineFaceLandmarks: false,
        minDetectionConfidence: 0.70,
        minTrackingConfidence: 0.70
      });
      
      this.holistic.onResults((results) => this.onBrowserFrame(results));
      
      // Initialize camera for browser
      if (this.videoElement) {
        if (typeof Camera === 'undefined') {
          const script = document.createElement('script');
          script.src = '/static/mediapipe/camera_utils/camera_utils.js';
          document.head.appendChild(script);
          await new Promise(r => script.onload = r);
        }
        
        this.cameraBrowser = new Camera(this.videoElement, {
          onFrame: async () => {
            if (this.isRunning && this.holistic) {
              await this.holistic.send({ image: this.videoElement });
            }
          },
          width: 480,
          height: 360
        });
      }
      
      return true;
    } catch (e) {
      console.error("[Camera] Browser init failed:", e);
      return false;
    }
  }
  
  /**
   * Handle frames from browser Mediapipe
   */
  onBrowserFrame(results) {
    if (!this.isRunning) return;
    
    const frameData = {
      timestamp: Date.now(),
      results: results,
      pose: results.poseLandmarks || [],
      left_hand: results.leftHandLandmarks || [],
      right_hand: results.rightHandLandmarks || [],
      face: results.faceLandmarks || []
    };
    
    this.addFrameToBuffer(frameData);
    
    if (this.onFrame) {
      this.onFrame(frameData);
    }
  }
  
  /**
   * Get frames from C++ backend
   */
  async fetchCPPFrame() {
    try {
      const response = await fetch("/camera/frame");
      if (response.status === 204) return null;  // No frame
      if (!response.ok) return null;
      
      const data = await response.json();
      if (data.status !== 'ok') return null;
      
      return {
        timestamp: data.timestamp * 1000,
        pose: data.pose,
        left_hand: data.left_hand,
        right_hand: data.right_hand,
        frame_b64: data.frame_b64
      };
    } catch (e) {
      console.warn("[Camera] Failed to fetch C++ frame:", e);
      return null;
    }
  }
  
  /**
   * Processing loop for C++ backend
   */
  async cppProcessingLoop() {
    while (this.isRunning && this.usingCPP) {
      const frame = await this.fetchCPPFrame();
      if (frame) {
        this.addFrameToBuffer(frame);
        
        if (this.onFrame) {
          this.onFrame(frame);
        }
        
        // Display frame on canvas if provided
        if (this.canvasElement && frame.frame_b64) {
          this.displayFrameOnCanvas(frame.frame_b64);
        }
      }
      
      await new Promise(r => setTimeout(r, 33));  // ~30fps
    }
  }
  
  /**
   * Display base64 frame on canvas
   */
  displayFrameOnCanvas(frameB64) {
    if (!this.canvasElement) return;
    
    const img = new Image();
    img.onload = () => {
      const ctx = this.canvasElement.getContext('2d');
      ctx.drawImage(img, 0, 0, this.canvasElement.width, this.canvasElement.height);
    };
    img.src = frameB64;
  }
  
  /**
   * Add frame to buffer
   */
  addFrameToBuffer(frame) {
    this.frameBuffer.push(frame);
    if (this.frameBuffer.length > this.maxBufferSize) {
      this.frameBuffer.shift();
    }
  }
  
  /**
   * Start camera capture
   */
  async start() {
    if (this.isRunning) return true;
    
    try {
      if (this.usingCPP) {
        // Start C++ camera
        const response = await fetch("/camera/start", { method: "POST" });
        if (!response.ok) {
          console.error("[Camera] Failed to start C++ backend");
          return false;
        }
        
        this.isRunning = true;
        
        // Start processing loop
        this.cppProcessingLoop();
      } else if (this.usingBrowser) {
        // Start browser camera
        if (!this.cameraBrowser) {
          console.error("[Camera] Browser camera not initialized");
          return false;
        }
        
        this.isRunning = true;
        await this.cameraBrowser.start();
      }
      
      console.log("[Camera] Started");
      return true;
    } catch (e) {
      console.error("[Camera] Start failed:", e);
      return false;
    }
  }
  
  /**
   * Stop camera capture
   */
  async stop() {
    if (!this.isRunning) return true;
    
    try {
      this.isRunning = false;
      
      if (this.usingCPP) {
        await fetch("/camera/stop", { method: "POST" });
      } else if (this.usingBrowser && this.cameraBrowser) {
        await this.cameraBrowser.stop();
      }
      
      console.log("[Camera] Stopped");
      return true;
    } catch (e) {
      console.error("[Camera] Stop failed:", e);
      return false;
    }
  }
  
  /**
   * Get latest frame data
   */
  getLatestFrame() {
    if (this.frameBuffer.length === 0) return null;
    return this.frameBuffer[this.frameBuffer.length - 1];
  }
  
  /**
   * Get all buffered frames
   */
  getAllFrames() {
    return [...this.frameBuffer];
  }
  
  /**
   * Clear frame buffer
   */
  clearFrameBuffer() {
    this.frameBuffer = [];
  }
  
  /**
   * Extract landmarks from buffered frames for model input
   * Matches the feature extraction logic from original JS
   */
  extractLandmarksSequence() {
    const sequences = [];
    
    for (const frame of this.frameBuffer) {
      const landmarks = [];
      
      // Collect all landmark coordinates
      if (frame.pose) {
        frame.pose.forEach(pt => {
          landmarks.push(pt[0] || pt.x || 0);
          landmarks.push(pt[1] || pt.y || 0);
          landmarks.push(pt[2] || pt.z || 0);
        });
      }
      
      if (frame.left_hand) {
        frame.left_hand.forEach(pt => {
          landmarks.push(pt[0] || pt.x || 0);
          landmarks.push(pt[1] || pt.y || 0);
          landmarks.push(pt[2] || pt.z || 0);
        });
      }
      
      if (frame.right_hand) {
        frame.right_hand.forEach(pt => {
          landmarks.push(pt[0] || pt.x || 0);
          landmarks.push(pt[1] || pt.y || 0);
          landmarks.push(pt[2] || pt.z || 0);
        });
      }
      
      if (landmarks.length > 0) {
        sequences.push(landmarks);
      }
    }
    
    return sequences;
  }
  
  /**
   * Check if using C++ backend
   */
  isCPP() {
    return this.usingCPP;
  }
  
  /**
   * Check if using browser backend
   */
  isBrowser() {
    return this.usingBrowser;
  }
  
  /**
   * Get status info
   */
  getStatus() {
    return {
      running: this.isRunning,
      backend: this.usingCPP ? 'cpp' : (this.usingBrowser ? 'browser' : 'none'),
      framesBuffered: this.frameBuffer.length,
      maxBufferSize: this.maxBufferSize
    };
  }
}

// ============================================
// Convenience exports
// ============================================
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CameraHandler;
}
