/**
 * VRM/GLTF Avatar Poser
 * 
 * Loads 3D models and poses them in real-time based on MediaPipe landmarks.
 * Used to replace the 2D skeleton feedback with a 3D avatar.
 * 
 * Usage:
 *   const poser = new AvatarPoser(containerElement, modelUrl);
 *   awaitposer.loadModel();
 *  poser.updateLandmarks(mediPipeResults); // Call with each frame's landmarks
 *   poser.dispose(); // Cleanup
 */

import * as THREE from '../vendor/three/three.module.js';
import { GLTFLoader } from '../vendor/three/examples/jsm/loaders/GLTFLoader.js';
// import { VRMLoader } from '../vendor/three-vrm.module.js'; // Uncomment if using VRM

class AvatarPoser {
  constructor(container, modelPath, options = {}) {
    this.container = container;
    this.modelPath = modelPath;
    this.options = {
      autoScale: true,
      autoCenter: true,
      damping: 0.3,
      ...options
    };

    // Three.js components
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.model = null;
    this.bones = {};
    this.animationId = null;

    // Landmark smoothing
    this.smoothedLandmarks = null;
    this.smoothingFactor = 0.4;

    // Canvas for fallback 2D skeleton
    this.canvas = null;
    this.ctx = null;
    this.is3DMode = true;
    this.use2DFallback = false;

    // MediaPipe connections for fallback
    this.POSE_CONNECTIONS = [
      [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
      [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
      [11, 23], [12, 24], [23, 24],
      [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32]
    ];
    this.HAND_CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],
      [0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]
    ];      // Bone mapping - customize this based on your 3D model's bone names
    this.boneMapping = {
      // Pose (body)
      'spine': ['Spine', 'Waist', 'Hips'],
      'leftShoulder': ['LeftShoulder', 'LeftArm'],
      'rightShoulder': ['RightShoulder', 'RightArm'],
      'leftElbow': ['LeftForeArm', 'LeftHand'],
      'rightElbow': ['RightForeArm', 'RightHand'],
      'leftWrist': ['LeftHand', 'LeftHandIndex1', 'LeftHandThumb4'],
      'rightWrist': ['RightHand', 'RightHandIndex1', 'RightHandThumb4'],
      // Head
      'head': ['Head', 'Neck'],
      'neck': ['Neck', 'Head'],
    };
  }

  async init() {
    if (this.is3DMode) {
      return this.init3D();
    } else {
      return this.init2D();
    }
  }

  async init3D() {
    // Create scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f172a);

    // Create camera
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
    this.camera.position.set(0, 1.2, 2.5);
    this.camera.lookAt(0, 1, 0);

    // Create renderer
    this.renderer = new THREE.WebGLRenderer({ 
      antialias: true,
      alpha: true 
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
    directionalLight.position.set(2, 4, 3);
    directionalLight.castShadow = true;
    this.scene.add(directionalLight);

    const fillLight = new THREE.DirectionalLight(0x4cc9f0, 0.4);
    fillLight.position.set(-2, 2, -2);
    this.scene.add(fillLight);

    // Ground plane for reference
    const groundGeo = new THREE.PlaneGeometry(4, 4);
    const groundMat = new THREE.MeshStandardMaterial({ 
      color: 0x1e293b,
      roughness: 0.8
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = 0;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Grid helper
    const gridHelper = new THREE.GridHelper(4, 20, 0x334155, 0x1e293b);
    this.scene.add(gridHelper);

    // Load model
    await this.loadModel();

    // Handle resize
    this.handleResize = this.handleResize.bind(this);
    window.addEventListener('resize', this.handleResize);

    // Start render loop
    this.animate();
  }

  async loadModel() {
    const loader = new GLTFLoader();
    
    return new Promise((resolve, reject) => {
      loader.load(
        this.modelPath,
        (gltf) => {
          this.model = gltf.scene;
          
          // Setup model
          this.model.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              // Make material look better
              if (child.material) {
                child.material.side = THREE.DoubleSide;
              }
            }
            // Store bone references
            if (child.isBone) {
              this.bones[child.name] = child;
            }
          });

          // Auto scale and center
          if (this.options.autoScale || this.options.autoCenter) {
            const box = new THREE.Box3().setFromObject(this.model);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            if (this.options.autoScale) {
              const maxDim = Math.max(size.x, size.y, size.z);
              const scale = 1.5 / maxDim;
              this.model.scale.setScalar(scale);
            }

            if (this.options.autoCenter) {
              this.model.position.x = -center.x * this.model.scale.x;
              this.model.position.y = -box.min.y * this.model.scale.y;
              this.model.position.z = -center.z * this.model.scale.z;
            }
          }

          this.scene.add(this.model);
          console.log('[AvatarPoser] Model loaded:', this.modelPath);
          resolve(this.model);
        },
        (progress) => {
          const percent = (progress.loaded / progress.total * 100).toFixed(0);
          console.log('[AvatarPoser] Loading:', percent + '%');
        },
        (error) => {
          console.error('[AvatarPoser] Load error:', error);
          // Fallback to 2D mode if 3D fails
          console.log('[AvatarPoser] Falling back to 2D mode');
          this.use2DFallback = true;
          this.init2D().then(resolve).catch(reject);
        }
      );
    });
  }

  init2D() {
    // Fallback to 2D canvas skeleton
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.container.clientWidth;
    this.canvas.height = this.container.clientHeight;
    this.ctx = this.canvas.getContext('2d');
    this.container.appendChild(this.canvas);
    this.use2DFallback = true;
    console.log('[AvatarPoser] Using 2D fallback mode');
    return Promise.resolve();
  }

  animate() {
    if (this.use2DFallback || !this.renderer) return;
    
    this.animationId = requestAnimationFrame(() => this.animate());
    
    if (this.model) {
      // Gentle idle animation
      const time = Date.now() * 0.001;
      this.model.rotation.y = Math.sin(time * 0.5) * 0.1;
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Update the avatar pose based on MediaPipe landmarks
   * @param {Object} landmarks - MediaPipe results containing poseLandmarks, leftHandLandmarks, rightHandLandmarks
   */
  updateLandmarks(landmarks) {
    if (!landmarks) return;

    // Smooth landmarks
    this.smoothedLandmarks = this.lerpLandmarks(
      this.smoothedLandmarks,
      landmarks,
      this.smoothingFactor
    );

    if (this.use2DFallback) {
      this.draw2DSkeleton(this.smoothedLandmarks);
    } else {
      this.pose3DModel(this.smoothedLandmarks);
    }
  }

  lerpLandmarks(prev, curr, factor) {
    if (!prev) return curr;
    if (!curr) return prev;

    const result = {};
    for (const key of ['poseLandmarks', 'faceLandmarks', 'leftHandLandmarks', 'rightHandLandmarks']) {
      if (!curr[key] || !prev[key]) {
        result[key] = curr[key] || prev[key];
        continue;
      }

      result[key] = curr[key].map((landmark, i) => {
        const prevLandmark = prev[key][i] || landmark;
        return {
          x: prevLandmark.x + (landmark.x - prevLandmark.x) * factor,
          y: prevLandmark.y + (landmark.y - prevLandmark.y) * factor,
          z: prevLandmark.z + (landmark.z - prevLandmark.z) * factor
        };
      });
    }
    return result;
  }

  /**
   * Pose the 3D model based on landmarks
   * This is a simplified version - real implementation requires 
   * custom bone rotation calculations based on your specific model
   */
  pose3DModel(landmarks) {
    if (!this.model || !landmarks.poseLandmarks) return;

    const pose = landmarks.poseLandmarks;
    const leftHand = landmarks.leftHandLandmarks;
    const rightHand = landmarks.rightHandLandmarks;

    // Map pose landmarks to body parts
    // MediaPipe pose landmarks: 0-32
    // 11: left shoulder, 12: right shoulder
    // 13: left elbow, 14: right elbow  
    // 15: left wrist, 16: right wrist
    // 23: left hip, 24: right hip

    try {
      // Get key landmarks
      const leftShoulder = pose[11];
      const rightShoulder = pose[12];
      const leftElbow = pose[13];
      const rightElbow = pose[14];
      const leftWrist = pose[15];
      const rightWrist = pose[16];
      const leftHip = pose[23];
      const rightHip = pose[24];
      const nose = pose[0];

      // Calculate body orientation
      if (leftShoulder && rightShoulder) {
        const shoulderDiff = rightShoulder.x - leftShoulder.x;
        const shoulderDist = Math.sqrt(
          Math.pow(rightShoulder.x - leftShoulder.x, 2) +
          Math.pow(rightShoulder.y - leftShoulder.y, 2)
        );
        
        // Rotate torso based on shoulder position
        if (this.bones['Spine'] || this.bones['Waist']) {
          const bodyRotation = Math.atan2(
            rightShoulder.y - leftShoulder.y,
            rightShoulder.x - leftShoulder.x
          );
          // Apply rotation to spine bones
          for (const boneName of ['Spine', 'Waist', 'Hips']) {
            if (this.bones[boneName]) {
              this.bones[boneName].rotation.z = -bodyRotation * 0.5;
            }
          }
        }

        // Position arms based on elbow/wrist
        this.poseArm('left', leftShoulder, leftElbow, leftWrist);
        this.poseArm('right', rightShoulder, rightElbow, rightWrist);
      }

      // Head/Neck orientation
      if (nose && leftShoulder && rightShoulder) {
        const neckPos = new THREE.Vector3(
          (leftShoulder.x + rightShoulder.x) / 2,
          (leftShoulder.y + rightShoulder.y) / 2,
          (leftShoulder.z + rightShoulder.z) / 2
        );

        const headAngleX = Math.atan2(nose.y - neckPos.y, nose.z - neckPos.z);
        const headAngleY = Math.atan2(nose.x - neckPos.x, nose.z - neckPos.z);

        for (const boneName of ['Head', 'Neck']) {
          if (this.bones[boneName]) {
            this.bones[boneName].rotation.x = headAngleX * 0.3;
            this.bones[boneName].rotation.y = -headAngleY * 0.5;
          }
        }
      }

      // Hand poses - simplified finger positioning
      this.poseHand('left', leftHand);
      this.poseHand('right', rightHand);

    } catch (e) {
      console.warn('[AvatarPoser] Posing error:', e);
    }
  }

  poseArm(side, shoulder, elbow, wrist) {
    if (!shoulder || !elbow || !wrist) return;

    const sidePrefix = side === 'left' ? 'Left' : 'Right';
    const oppositePrefix = side === 'left' ? 'Right' : 'Left';

    // Calculate arm angles
    const upperArmDir = {
      x: elbow.x - shoulder.x,
      y: elbow.y - shoulder.y,
      z: elbow.z - shoulder.z
    };
    
    const lowerArmDir = {
      x: wrist.x - elbow.x,
      y: wrist.y - elbow.y,
      z: wrist.z - elbow.z
    };

    // Upper arm rotation (shoulder to elbow)
    const upperArmAngleX = Math.atan2(
      Math.sqrt(upperArmDir.x * upperArmDir.x + upperArmDir.z * upperArmDir.z),
      upperArmDir.y
    ) - Math.PI / 2;
    
    const upperArmAngleZ = Math.atan2(upperArmDir.y, upperArmDir.z);

    // Apply to shoulder/arm bones
    const armBoneName = `${sidePrefix}Arm`;
    const forearmBoneName = `${sidePrefix}ForeArm`;
    
    if (this.bones[armBoneName]) {
      this.bones[armBoneName].rotation.x = upperArmAngleX * 0.5;
      this.bones[armBoneName].rotation.z = side === 'left' ? -upperArmAngleZ * 0.3 : upperArmAngleZ * 0.3;
    }

    // Lower arm rotation (elbow to wrist)
    if (this.bones[forearmBoneName]) {
      const forearmAngleX = Math.atan2(
        Math.sqrt(lowerArmDir.x * lowerArmDir.x + lowerArmDir.z * lowerArmDir.z),
        lowerArmDir.y
      ) - Math.PI / 2;
      this.bones[forearmBoneName].rotation.x = forearmAngleX * 0.3;
    }
  }

  poseHand(side, handLandmarks) {
    if (!handLandmarks || handLandmarks.length < 21) return;

    const sidePrefix = side === 'left' ? 'Left' : 'Right';

    // Calculate average hand position for gross positioning
    let avgX = 0, avgY = 0, avgZ = 0;
    handLandmarks.forEach(lm => {
      avgX += lm.x;
      avgY += lm.y;
      avgZ += lm.z;
    });
    avgX /= handLandmarks.length;
    avgY /= handLandmarks.length;
    avgZ /= handLandmarks.length;

    // Finger spread calculation (thumb to pinky)
    const thumbTip = handLandmarks[4];
    const pinkyTip = handLandmarks[20];
    const fingerSpread = Math.sqrt(
      Math.pow(pinkyTip.x - thumbTip.x, 2) +
      Math.pow(pinkyTip.y - thumbTip.y, 2)
    );

    // Apply to hand bone
    const handBoneName = `${sidePrefix}Hand`;
    if (this.bones[handBoneName]) {
      // Simple hand pose - spread fingers based on detected spread
      this.bones[handBoneName].rotation.x = (avgY - 0.5) * 0.5;
      this.bones[handBoneName].rotation.y = (0.5 - avgX) * 0.3;
    }

    // Try to pose individual fingers if bones exist
    const fingerNames = ['Index', 'Middle', 'Ring', 'Pinky', 'Thumb'];
    const tipIndices = [8, 12, 16, 20, 4];
    const baseIndices = [5, 9, 13, 17, 1];

    fingerNames.forEach((fingerName, i) => {
      const tip = handLandmarks[tipIndices[i]];
      const base = handLandmarks[baseIndices[i]];
      
      if (tip && base) {
        const curl = Math.atan2(tip.y - base.y, tip.z - base.z);
        
        // Try to find finger bones
        for (let j = 1; j <= 3; j++) {
          const boneName = `${sidePrefix}${fingerName}${j}`;
          if (this.bones[boneName]) {
            this.bones[boneName].rotation.x = curl * 0.3;
          }
        }
      }
    });
  }

  /**
   * Draw 2D skeleton fallback
   */
  draw2DSkeleton(landmarks) {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    const drawPoint = (point, color, radius = 6) => {
      if (!point) return;
      const x = point.x * w;
      const y = point.y * h;
      
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    };

    const drawLine = (start, end, color, width = 4) => {
      if (!start || !end) return;
      ctx.beginPath();
      ctx.moveTo(start.x * w, start.y * h);
      ctx.lineTo(end.x * w, end.y * h);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();
    };

    // Draw pose connections
    if (landmarks.poseLandmarks) {
      ctx.shadowBlur = 12;
      ctx.shadowColor = 'rgba(167, 139, 250, 0.6)';
      
      this.POSE_CONNECTIONS.forEach(([start, end]) => {
        const startPoint = landmarks.poseLandmarks[start];
        const endPoint = landmarks.poseLandmarks[end];
        if (startPoint && endPoint) {
          drawLine(startPoint, endPoint, 'rgba(167, 139, 250, 0.9)');
        }
      });

      // Draw pose landmarks
      landmarks.poseLandmarks.forEach(lm => {
        drawPoint(lm, 'rgba(167, 139, 250, 1)');
      });
    }

    // Draw face (sparse)
    if (landmarks.faceLandmarks) {
      landmarks.faceLandmarks.forEach((lm, i) => {
        if (i % 10 === 0) {
          drawPoint(lm, 'rgba(255, 140, 66, 0.8)', 3);
        }
      });
    }

    // Draw hands
    const drawHand = (handLandmarks, color) => {
      if (!handLandmarks || handLandmarks.length === 0) return;

      this.HAND_CONNECTIONS.forEach(([start, end]) => {
        const startPoint = handLandmarks[start];
        const endPoint = handLandmarks[end];
        if (startPoint && endPoint) {
          drawLine(startPoint, endPoint, color);
        }
      });

      handLandmarks.forEach(lm => {
        drawPoint(lm, color, 5);
      });
    };

    if (landmarks.rightHandLandmarks) {
      drawHand(landmarks.rightHandLandmarks, 'rgba(76, 201, 240, 0.95)');
    }
    if (landmarks.leftHandLandmarks) {
      drawHand(landmarks.leftHandLandmarks, 'rgba(6, 214, 160, 0.95)');
    }

    ctx.shadowBlur = 0;
  }

  handleResize() {
    if (this.use2DFallback) {
      this.canvas.width = this.container.clientWidth;
      this.canvas.height = this.container.clientHeight;
      return;
    }

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  dispose() {
    // Cancel animation
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }

    // Remove resize listener
    window.removeEventListener('resize', this.handleResize);

    // Dispose Three.js resources
    if (this.renderer) {
      this.renderer.dispose();
      this.container.removeChild(this.renderer.domElement);
    }

    // Clear scene
    if (this.scene) {
      this.scene.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          if (Array.isArray(object.material)) {
            object.material.forEach(m => m.dispose());
          } else {
            object.material.dispose();
          }
        }
      });
    }

    this.model = null;
    this.bones = {};

    // Remove canvas if in 2D mode
    if (this.canvas && this.container.contains(this.canvas)) {
      this.container.removeChild(this.canvas);
    }

    console.log('[AvatarPoser] Disposed');
  }

  /**
   * Set the model path and reload
   */
  async setModel(path) {
    this.modelPath = path;
    this.dispose();
    await this.init();
  }

  /**
   * Switch between 3D and 2D mode
   */
  async setMode(mode) {
    if (mode === '3d') {
      this.is3DMode = true;
    } else {
      this.is3DMode = false;
    }
    this.dispose();
    await this.init();
  }
}

// Export for use in other files
window.AvatarPoser = AvatarPoser;
export default AvatarPoser;