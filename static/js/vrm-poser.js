/**
 * VRM/GLTF Avatar Poser
 * 
 * Loads 3D models and poses them in real-time based on MediaPipe landmarks.
 * Used to replace the 2D skeleton feedback with a 3D avatar.
 * 
 * Usage:
 *   const poser = new AvatarPoser(containerElement, modelUrl);
 *   await poser.init();
 *   poser.updateLandmarks(mediaPipeResults); // Call with each frame's landmarks
 *   poser.dispose(); // Cleanup
 */

import * as THREE from '../vendor/three/three.module.js';
import { GLTFLoader } from '../vendor/three/examples/jsm/loaders/GLTFLoader.js';

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
    ];

    // Bone name mapping for the Blender model
    // Spine chain: spine -> spine.001 -> spine.002 -> spine.003 (chest)
    //              spine.003 -> spine.004 -> spine.005 -> spine.006 -> face
    // Arms:  shoulder.L/R -> upper_arm.L/R -> forearm.L/R -> hand.L/R
    // Fingers: palm.0X.L/R -> f_index/f_middle/f_ring/f_pinky.0X.L/R
    //          palm.01.L/R -> thumb.0X.L/R
    this.BONE_NAMES = {
      // Spine/torso
      hips: 'spine',
      spine1: 'spine.001',
      spine2: 'spine.002',
      chest: 'spine.003',
      // Neck/head
      neck1: 'spine.004',
      neck2: 'spine.005',
      neck3: 'spine.006',
      head: 'face',
      // Left arm
      shoulderL: 'shoulder.L',
      upperArmL: 'upper_arm.L',
      forearmL: 'forearm.L',
      handL: 'hand.L',
      // Right arm
      shoulderR: 'shoulder.R',
      upperArmR: 'upper_arm.R',
      forearmR: 'forearm.R',
      handR: 'hand.R',
      // Left hand fingers
      thumbL1: 'thumb.01.L',
      thumbL2: 'thumb.02.L',
      thumbL3: 'thumb.03.L',
      indexL1: 'f_index.01.L',
      indexL2: 'f_index.02.L',
      indexL3: 'f_index.03.L',
      middleL1: 'f_middle.01.L',
      middleL2: 'f_middle.02.L',
      middleL3: 'f_middle.03.L',
      ringL1: 'f_ring.01.L',
      ringL2: 'f_ring.02.L',
      ringL3: 'f_ring.03.L',
      pinkyL1: 'f_pinky.01.L',
      pinkyL2: 'f_pinky.02.L',
      pinkyL3: 'f_pinky.03.L',
      // Right hand fingers
      thumbR1: 'thumb.01.R',
      thumbR2: 'thumb.02.R',
      thumbR3: 'thumb.03.R',
      indexR1: 'f_index.01.R',
      indexR2: 'f_index.02.R',
      indexR3: 'f_index.03.R',
      middleR1: 'f_middle.01.R',
      middleR2: 'f_middle.02.R',
      middleR3: 'f_middle.03.R',
      ringR1: 'f_ring.01.R',
      ringR2: 'f_ring.02.R',
      ringR3: 'f_ring.03.R',
      pinkyR1: 'f_pinky.01.R',
      pinkyR2: 'f_pinky.02.R',
      pinkyR3: 'f_pinky.03.R',
    };

    // Store initial bone rotations (rest pose) so we can apply deltas
    this.restPose = {};
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
              if (child.material) {
                child.material.side = THREE.DoubleSide;
              }
            }
            // Store bone references
            if (child.isBone) {
              this.bones[child.name] = child;
            }
          });

          // Log all found bones for debugging
          const boneNames = Object.keys(this.bones);
          console.log('[AvatarPoser] Found', boneNames.length, 'bones:', boneNames);

          // Verify which expected bones were found
          const missing = [];
          const found = [];
          for (const [key, boneName] of Object.entries(this.BONE_NAMES)) {
            if (this.bones[boneName]) {
              found.push(boneName);
            } else {
              missing.push(boneName);
            }
          }
          console.log('[AvatarPoser] Matched bones:', found);
          if (missing.length > 0) {
            console.warn('[AvatarPoser] Missing expected bones:', missing);
          }

          // Store rest pose rotations
          for (const [key, boneName] of Object.entries(this.BONE_NAMES)) {
            const bone = this.bones[boneName];
            if (bone) {
              this.restPose[boneName] = {
                x: bone.rotation.x,
                y: bone.rotation.y,
                z: bone.rotation.z
              };
            }
          }

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
          if (progress.total > 0) {
            const percent = (progress.loaded / progress.total * 100).toFixed(0);
            console.log('[AvatarPoser] Loading:', percent + '%');
          }
        },
        (error) => {
          console.error('[AvatarPoser] Load error:', error);
          console.log('[AvatarPoser] Falling back to 2D mode');
          this.use2DFallback = true;
          this.init2D().then(resolve).catch(reject);
        }
      );
    });
  }

  init2D() {
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
    
    if (this.model && !this._isPosing) {
      const time = Date.now() * 0.001;
      this.model.rotation.y = Math.sin(time * 0.5) * 0.1;
    }

    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Helper to get a bone by its logical name from BONE_NAMES
   */
  getBone(logicalName) {
    const boneName = this.BONE_NAMES[logicalName];
    return boneName ? this.bones[boneName] : null;
  }

  /**
   * Helper to get the rest pose for a bone
   */
  getRest(logicalName) {
    const boneName = this.BONE_NAMES[logicalName];
    return boneName ? this.restPose[boneName] : null;
  }

  /**
   * Update the avatar pose based on MediaPipe landmarks
   */
  updateLandmarks(landmarks) {
    if (!landmarks) return;

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
   * Pose the 3D model based on MediaPipe landmarks.
   *
   * MediaPipe pose landmark indices:
   *   0: nose, 11: left shoulder, 12: right shoulder,
   *   13: left elbow, 14: right elbow,
   *   15: left wrist, 16: right wrist,
   *   23: left hip, 24: right hip
   */
  pose3DModel(landmarks) {
    if (!this.model || !landmarks.poseLandmarks) return;

    const pose = landmarks.poseLandmarks;
    const leftHand = landmarks.leftHandLandmarks;
    const rightHand = landmarks.rightHandLandmarks;

    this._isPosing = true;

    try {
    const leftShoulder = pose[11];
    const rightShoulder = pose[12];
    const leftElbow = pose[13];
    const rightElbow = pose[14];
    const leftWrist = pose[15];
    const rightWrist = pose[16];
    const nose = pose[0];

    // --- Torso / Spine rotation ---
    if (leftShoulder && rightShoulder) {
      const bodyTilt = Math.atan2(
        rightShoulder.y - leftShoulder.y,
        rightShoulder.x - leftShoulder.x
      );

      // Apply subtle rotation to spine bones
      const spineBones = ['spine1', 'spine2', 'chest'];
      spineBones.forEach(name => {
        const bone = this.getBone(name);
        const rest = this.getRest(name);
        if (bone && rest) {
          bone.rotation.z = rest.z - bodyTilt * 0.3;
        }
      });
    }

    // --- Head / Neck orientation ---
    if (nose && leftShoulder && rightShoulder) {
      const neckX = (leftShoulder.x + rightShoulder.x) / 2;
      const neckY = (leftShoulder.y + rightShoulder.y) / 2;
      const neckZ = (leftShoulder.z + rightShoulder.z) / 2;

      const headAngleX = Math.atan2(nose.y - neckY, nose.z - neckZ);
      const headAngleY = Math.atan2(nose.x - neckX, nose.z - neckZ);

      // Distribute rotation across neck bones and head
      const neckBones = ['neck1', 'neck2', 'neck3'];
      neckBones.forEach(name => {
        const bone = this.getBone(name);
        const rest = this.getRest(name);
        if (bone && rest) {
          bone.rotation.x = rest.x + headAngleX * 0.15;
          bone.rotation.y = rest.y - headAngleY * 0.2;
        }
      });

      const headBone = this.getBone('head');
      const headRest = this.getRest('head');
      if (headBone && headRest) {
        headBone.rotation.x = headRest.x + headAngleX * 0.2;
        headBone.rotation.y = headRest.y - headAngleY * 0.3;
      }
    }

    // --- Arms ---
    this.poseArm('L', leftShoulder, leftElbow, leftWrist);
    this.poseArm('R', rightShoulder, rightElbow, rightWrist);

    // --- Hands / Fingers ---
    this.poseHand('L', leftHand);
    this.poseHand('R', rightHand);

    } catch (e) {
      console.warn('[AvatarPoser] Posing error:', e);
    }
  }

  /**
   * Pose an arm chain: shoulder -> upper_arm -> forearm -> hand
   * @param {string} side - 'L' or 'R'
   */
  poseArm(side, shoulder, elbow, wrist) {
    if (!shoulder || !elbow || !wrist) return;

    const upperArmKey = side === 'L' ? 'upperArmL' : 'upperArmR';
    const forearmKey = side === 'L' ? 'forearmL' : 'forearmR';
    const handKey = side === 'L' ? 'handL' : 'handR';
    const shoulderKey = side === 'L' ? 'shoulderL' : 'shoulderR';

    // Direction vectors
    const upperArmDir = {
      x: elbow.x - shoulder.x,
      y: elbow.y - shoulder.y,
      z: (elbow.z || 0) - (shoulder.z || 0)
    };
    
    const forearmDir = {
      x: wrist.x - elbow.x,
      y: wrist.y - elbow.y,
      z: (wrist.z || 0) - (elbow.z || 0)
    };

    // Upper arm: angle from shoulder to elbow
    const upperArmAngleZ = Math.atan2(upperArmDir.y, upperArmDir.x);
    const upperArmLen = Math.sqrt(upperArmDir.x ** 2 + upperArmDir.y ** 2 + upperArmDir.z ** 2);
    const upperArmAngleX = Math.asin(Math.max(-1, Math.min(1, upperArmDir.z / (upperArmLen || 1))));

    const shoulderBone = this.getBone(shoulderKey);
    const shoulderRest = this.getRest(shoulderKey);
    if (shoulderBone && shoulderRest) {
      // Slight shoulder raise/drop
      shoulderBone.rotation.z = shoulderRest.z + (side === 'L' ? -1 : 1) * upperArmAngleZ * 0.15;
    }

    const upperArmBone = this.getBone(upperArmKey);
    const upperArmRest = this.getRest(upperArmKey);
    if (upperArmBone && upperArmRest) {
      // Main arm swing
      upperArmBone.rotation.z = upperArmRest.z + (side === 'L'
        ? (upperArmAngleZ + Math.PI / 2) * 0.6
        : (upperArmAngleZ - Math.PI / 2) * 0.6);
      upperArmBone.rotation.x = upperArmRest.x - upperArmAngleX * 0.5;
    }

    // Forearm: angle from elbow to wrist
    const forearmAngleZ = Math.atan2(forearmDir.y, forearmDir.x);
    const elbowBend = forearmAngleZ - upperArmAngleZ;

    const forearmBone = this.getBone(forearmKey);
    const forearmRest = this.getRest(forearmKey);
    if (forearmBone && forearmRest) {
      forearmBone.rotation.z = forearmRest.z + elbowBend * 0.5;
    }

    // Wrist/hand orientation
    const handBone = this.getBone(handKey);
    const handRest = this.getRest(handKey);
    if (handBone && handRest) {
      const wristAngle = Math.atan2(forearmDir.y, forearmDir.x);
      handBone.rotation.z = handRest.z + wristAngle * 0.2;
    }
  }

  /**
   * Pose fingers based on MediaPipe hand landmarks.
   *
   * MediaPipe hand landmark indices:
   *   0: wrist
   *   1-4: thumb (CMC, MCP, IP, TIP)
   *   5-8: index (MCP, PIP, DIP, TIP)
   *   9-12: middle (MCP, PIP, DIP, TIP)
   *   13-16: ring (MCP, PIP, DIP, TIP)
   *   17-20: pinky (MCP, PIP, DIP, TIP)
   *
   * @param {string} side - 'L' or 'R'
   */
  poseHand(side, handLandmarks) {
    if (!handLandmarks || handLandmarks.length < 21) return;

    const suffix = side;

    // Finger definitions: [boneName prefix, base landmark index]
    // Each finger has 3 bones mapping to 3 segments between 4 landmarks
    const fingers = [
      { prefix: 'thumb',  boneKeys: [`thumb${suffix}1`, `thumb${suffix}2`, `thumb${suffix}3`],  landmarks: [1, 2, 3, 4] },
      { prefix: 'index',  boneKeys: [`index${suffix}1`, `index${suffix}2`, `index${suffix}3`],  landmarks: [5, 6, 7, 8] },
      { prefix: 'middle', boneKeys: [`middle${suffix}1`, `middle${suffix}2`, `middle${suffix}3`], landmarks: [9, 10, 11, 12] },
      { prefix: 'ring',   boneKeys: [`ring${suffix}1`, `ring${suffix}2`, `ring${suffix}3`],   landmarks: [13, 14, 15, 16] },
      { prefix: 'pinky',  boneKeys: [`pinky${suffix}1`, `pinky${suffix}2`, `pinky${suffix}3`],  landmarks: [17, 18, 19, 20] },
    ];

    fingers.forEach(finger => {
      const lm = finger.landmarks.map(i => handLandmarks[i]);
      if (!lm[0] || !lm[1] || !lm[2] || !lm[3]) return;

      // For each bone segment, calculate curl angle
      for (let i = 0; i < 3; i++) {
        const bone = this.getBone(finger.boneKeys[i]);
        const rest = this.getRest(finger.boneKeys[i]);
        if (!bone || !rest) continue;

        // Direction of this segment vs next segment
        const segDir = {
          x: lm[i + 1].x - lm[i].x,
          y: lm[i + 1].y - lm[i].y
        };

        // For curl, use the angle of the segment relative to the wrist-to-base direction
        const curl = Math.atan2(segDir.y, segDir.x);
        bone.rotation.z = rest.z + curl * 0.3;
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
      if (this.canvas) {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = this.container.clientHeight;
      }
      return;
    }

    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  dispose() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = null;
    }

    window.removeEventListener('resize', this.handleResize);

    if (this.renderer) {
      this.renderer.dispose();
      if (this.container.contains(this.renderer.domElement)) {
        this.container.removeChild(this.renderer.domElement);
      }
    }

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
    this.restPose = {};

    if (this.canvas && this.container.contains(this.canvas)) {
      this.container.removeChild(this.canvas);
    }

    this.renderer = null;
    this.scene = null;
    this.camera = null;

    console.log('[AvatarPoser] Disposed');
  }

  async setModel(path) {
    this.modelPath = path;
    this.dispose();
    await this.init();
  }

  async setMode(mode) {
    if (mode === '3d') {
      this.is3DMode = true;
      this.use2DFallback = false;
    } else {
      this.is3DMode = false;
    }
    this.dispose();
    await this.init();
  }
}

window.AvatarPoser = AvatarPoser;
export default AvatarPoser;
