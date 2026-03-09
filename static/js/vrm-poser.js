/**
 * VRM/GLTF Avatar Poser
 *
 * Loads 3D models and poses them in real-time based on MediaPipe landmarks.
 * Used to replace the 2D skeleton feedback with a 3D avatar.
 *
 * Usage:
 *   const poser = new AvatarPoser(containerElement, modelUrl);
 *   await poser.init();
 *   poser.updateLandmarks(mediaPipeResults);
 *   poser.dispose();
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

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.model = null;
    this.bones = {};
    this.animationId = null;

    this.smoothedLandmarks = null;
    this.smoothingFactor = 0.4;

    this.canvas = null;
    this.ctx = null;
    this.is3DMode = true;
    this.use2DFallback = false;
    this._isPosing = false;

    // Rest pose data
    this.restPose = {};                  // euler {x,y,z} per bone (for spine/head/finger posing)
    this.restQuaternions = {};            // THREE.Quaternion per bone (local rest rotation)
    this.boneWorldRestQuaternions = {};   // THREE.Quaternion per bone (world rest rotation)
    this.boneRestDirections = {};         // THREE.Vector3 per bone (world-space direction to child)

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

    this.BONE_NAMES = {
      hips: 'spine',
      spine1: 'spine001',
      spine2: 'spine002',
      spine3: 'spine003',
      spine4: 'spine004',
      spine5: 'spine005',
      spine6: 'spine006',
      chest: 'spine003',
      neck1: 'spine004',
      neck2: 'spine005',
      neck3: 'spine006',
      head: 'face',
      nose: 'nose',
      jaw: 'jaw',
      chin: 'chin',
      shoulderL: 'shoulderL',
      upperArmL: 'upper_armL',
      forearmL: 'forearmL',
      handL: 'handL',
      palmL1: 'palm01L',
      palmL2: 'palm02L',
      palmL3: 'palm03L',
      palmL4: 'palm04L',
      thumbL1: 'thumb01L',
      thumbL2: 'thumb02L',
      thumbL3: 'thumb03L',
      indexL1: 'f_index01L',
      indexL2: 'f_index02L',
      indexL3: 'f_index03L',
      middleL1: 'f_middle01L',
      middleL2: 'f_middle02L',
      middleL3: 'f_middle03L',
      ringL1: 'f_ring01L',
      ringL2: 'f_ring02L',
      ringL3: 'f_ring03L',
      pinkyL1: 'f_pinky01L',
      pinkyL2: 'f_pinky02L',
      pinkyL3: 'f_pinky03L',
      shoulderR: 'shoulderR',
      upperArmR: 'upper_armR',
      forearmR: 'forearmR',
      handR: 'handR',
      palmR1: 'palm01R',
      palmR2: 'palm02R',
      palmR3: 'palm03R',
      palmR4: 'palm04R',
      thumbR1: 'thumb01R',
      thumbR2: 'thumb02R',
      thumbR3: 'thumb03R',
      indexR1: 'f_index01R',
      indexR2: 'f_index02R',
      indexR3: 'f_index03R',
      middleR1: 'f_middle01R',
      middleR2: 'f_middle02R',
      middleR3: 'f_middle03R',
      ringR1: 'f_ring01R',
      ringR2: 'f_ring02R',
      ringR3: 'f_ring03R',
      pinkyR1: 'f_pinky01R',
      pinkyR2: 'f_pinky02R',
      pinkyR3: 'f_pinky03R',
      breastL: 'breastL',
      breastR: 'breastR',
      pelvisL: 'pelvisL',
      pelvisR: 'pelvisR',
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
    try {
      console.log('[AvatarPoser] init3D: Starting 3D initialization');
      this.scene = new THREE.Scene();
      this.scene.background = new THREE.Color(0x0f172a);

      const aspect = this.container.clientWidth / this.container.clientHeight;
      this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
      this.camera.position.set(0, 1.2, 2.5);
      this.camera.lookAt(0, 1, 0);

      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      this.renderer.shadowMap.enabled = true;
      this.renderer.outputColorSpace = THREE.SRGBColorSpace;
      this.container.appendChild(this.renderer.domElement);

      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      this.scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1.2);
      directionalLight.position.set(2, 4, 3);
      directionalLight.castShadow = true;
      this.scene.add(directionalLight);

      console.log('[AvatarPoser] init3D: Scene, camera, renderer, and lights set up');
    } catch (e) {
      console.error('[AvatarPoser] init3D: Error during initialization', e);
    }
  }

  async loadModel() {
    const loader = new GLTFLoader();

    return new Promise((resolve, reject) => {
      loader.load(
        this.modelPath,
        (gltf) => {
          this.model = gltf.scene;

          this.model.traverse((child) => {
            if (child.isMesh) {
              child.castShadow = true;
              child.receiveShadow = true;
              if (child.material) {
                child.material.side = THREE.DoubleSide;
              }
            }
            if (child.isBone) {
              this.bones[child.name] = child;
              if (!window._allBoneNames) window._allBoneNames = [];
              window._allBoneNames.push(child.name);
            }
          });

          const boneNames = Object.keys(this.bones);
          console.log('[AvatarPoser] Found', boneNames.length, 'bones:', boneNames);
          console.log('[AvatarPoser] All bone names in model:', window._allBoneNames);

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

          // Rotate model to face the camera
          this.model.rotation.y = Math.PI;

          this.scene.add(this.model);

          // Compute and store rest pose data AFTER positioning/rotating
          this.model.updateMatrixWorld(true);
          this._storeRestPoseData();

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
          this.use2DFallback = true;
          this.init2D().then(resolve).catch(reject);
        }
      );
    });
  }

  _storeRestPoseData() {
    for (const [key, boneName] of Object.entries(this.BONE_NAMES)) {
      const bone = this.bones[boneName];
      if (!bone) continue;

      // Euler rest pose (for spine/head/finger posing)
      this.restPose[boneName] = {
        x: bone.rotation.x,
        y: bone.rotation.y,
        z: bone.rotation.z
      };

      // Local rest quaternion
      this.restQuaternions[boneName] = bone.quaternion.clone();

      // World rest quaternion
      const worldQuat = new THREE.Quaternion();
      bone.getWorldQuaternion(worldQuat);
      this.boneWorldRestQuaternions[boneName] = worldQuat;

      // Rest direction: bone → first bone child (in world space)
      const boneChild = bone.children.find(c => c.isBone);
      if (boneChild) {
        const bonePos = new THREE.Vector3();
        const childPos = new THREE.Vector3();
        bone.getWorldPosition(bonePos);
        boneChild.getWorldPosition(childPos);
        const dir = new THREE.Vector3().subVectors(childPos, bonePos);
        if (dir.lengthSq() > 0.0001) {
          this.boneRestDirections[boneName] = dir.normalize();
        }
      }
    }

    const dirCount = Object.keys(this.boneRestDirections).length;
    console.log('[AvatarPoser] Stored rest directions for', dirCount, 'bones');
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

    // Subtle idle sway when not actively posing
    if (this.model && !this._isPosing) {
      const time = Date.now() * 0.001;
      this.model.rotation.y = Math.PI + Math.sin(time * 0.5) * 0.1;
    }

    this.renderer.render(this.scene, this.camera);
  }

  getBone(logicalName) {
    const boneName = this.BONE_NAMES[logicalName];
    return boneName ? this.bones[boneName] : null;
  }

  getRest(logicalName) {
    const boneName = this.BONE_NAMES[logicalName];
    return boneName ? this.restPose[boneName] : null;
  }

  // ─── Landmark processing ─────────────────────────────────────────

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

  // ─── Quaternion-based posing helpers ──────────────────────────────

  /**
   * Compute a world-space direction vector from two MediaPipe landmarks.
   *
   * MediaPipe coordinate system:
   *   x: 0→1 left→right on screen
   *   y: 0→1 top→bottom on screen
   *   z: depth (smaller = closer to camera)
   *
   * Three.js world (camera at +Z looking at -Z):
   *   +X = screen right, +Y = up, -Z = into screen
   *
   * Mapping: worldX = lm.x, worldY = -lm.y, worldZ = -lm.z
   */
  _landmarkDir(from, to) {
    const dx = to.x - from.x;
    const dy = -(to.y - from.y);
    const dz = -((to.z || 0) - (from.z || 0));
    const lenSq = dx * dx + dy * dy + dz * dz;
    if (lenSq < 0.00001) return null;
    const len = Math.sqrt(lenSq);
    return new THREE.Vector3(dx / len, dy / len, dz / len);
  }

  /**
   * Reset all mapped bones to their rest-pose quaternions.
   */
  _resetToRestPose() {
    for (const boneName of Object.values(this.BONE_NAMES)) {
      const bone = this.bones[boneName];
      const restQ = this.restQuaternions[boneName];
      if (bone && restQ) {
        bone.quaternion.copy(restQ);
      }
    }
    this.model.updateMatrixWorld(true);
  }

  /**
   * Pose a bone so it points toward `targetWorldDir` (world space).
   *
   * Uses Quaternion.setFromUnitVectors to compute the rotation delta
   * between the bone's rest direction and the target direction,
   * then converts to the bone's local space.
   *
   * @param {string} boneName - e.g. 'upper_arm.L'
   * @param {THREE.Vector3} targetWorldDir - normalised target direction in world space
   * @param {number} blend - 0..1, how much to apply (1 = full, 0 = rest)
   */
  _poseBoneToDirection(boneName, targetWorldDir, blend = 1.0) {
    const bone = this.bones[boneName];
    if (!bone) return;

    const restDir = this.boneRestDirections[boneName];
    if (!restDir) return;

    const restWorldQuat = this.boneWorldRestQuaternions[boneName];
    if (!restWorldQuat) return;

    // Guard against nearly-opposite directions (setFromUnitVectors is unstable there)
    const dot = restDir.dot(targetWorldDir);
    if (dot < -0.95) return;

    // Delta rotation in world space: rotates restDir → targetWorldDir
    const deltaQuat = new THREE.Quaternion().setFromUnitVectors(restDir, targetWorldDir);

    // Target world quaternion = delta applied to rest world orientation
    const targetWorldQuat = new THREE.Quaternion().multiplyQuaternions(deltaQuat, restWorldQuat);

    // Convert to bone-local: localQuat = inverse(parentCurrentWorld) * targetWorld
    const parentWorldQuat = new THREE.Quaternion();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(parentWorldQuat);
    }
    const localQuat = new THREE.Quaternion()
      .copy(parentWorldQuat)
      .invert()
      .multiply(targetWorldQuat);

    // Blend between rest and target
    if (blend < 1.0) {
      const restLocal = this.restQuaternions[boneName];
      if (restLocal) {
        bone.quaternion.copy(restLocal).slerp(localQuat, blend);
      } else {
        bone.quaternion.copy(localQuat);
      }
    } else {
      bone.quaternion.copy(localQuat);
    }
  }

  // ─── Main 3D posing entry point ─────────────────────────────────

  /**
   * Pose the 3D model from MediaPipe landmark data.
   *
   * Processing order matters (parent bones before children):
   *   1. Reset all bones to rest pose
   *   2. Spine / torso  (euler)
   *   3. Head / neck     (euler)
   *   4. Arms            (quaternion — parent world quats now include spine changes)
   *   5. Fingers         (euler curl)
   */
  pose3DModel(landmarks) {
    if (!this.model || !landmarks.poseLandmarks) {
      console.warn('[AvatarPoser] pose3DModel: Model or poseLandmarks missing');
      return;
    }

    const pose = landmarks.poseLandmarks;
    const leftHand = landmarks.leftHandLandmarks;
    const rightHand = landmarks.rightHandLandmarks;

    this._isPosing = true;

    try {
      // 1. Reset to rest pose so each frame starts clean
      this._resetToRestPose();
      console.log('[AvatarPoser] pose3DModel: Rest pose reset');

      const leftShoulder  = pose[11];
      const rightShoulder = pose[12];
      const leftElbow     = pose[13];
      const rightElbow    = pose[14];
      const leftWrist     = pose[15];
      const rightWrist    = pose[16];
      const nose          = pose[0];

      // 2. Spine / torso (euler-based tilt from shoulder line)
      this._poseSpine(leftShoulder, rightShoulder);
      // 3. Head / neck
      this._poseHead(nose, leftShoulder, rightShoulder);
      // 4. Arms (quaternion-based direction alignment)
      this._poseArmQ('L', leftShoulder, leftElbow, leftWrist, leftHand);
      this._poseArmQ('R', rightShoulder, rightElbow, rightWrist, rightHand);
      // 5. Fingers
      this._poseFingers('L', leftHand);
      this._poseFingers('R', rightHand);
      // Keep model facing the camera
      this.model.rotation.y = Math.PI;
      console.log('[AvatarPoser] pose3DModel: Pose applied and model rotated');
    } catch (e) {
      console.error('[AvatarPoser] Posing error:', e);
    }
  }

  // ─── Spine ───────────────────────────────────────────────────────

  _poseSpine(leftShoulder, rightShoulder) {
    if (!leftShoulder || !rightShoulder) {
      console.warn('[AvatarPoser] _poseSpine: Missing shoulders', { leftShoulder, rightShoulder });
      return;
    }

    // Body tilt from shoulder line angle
    const bodyTilt = Math.atan2(
      rightShoulder.y - leftShoulder.y,
      rightShoulder.x - leftShoulder.x
    );

    const spineBones = ['spine1', 'spine2', 'chest'];
    spineBones.forEach(name => {
      const bone = this.getBone(name);
      const rest = this.getRest(name);
      if (!bone) {
        console.warn(`[AvatarPoser] _poseSpine: Bone not found: ${name}`);
      }
      if (!rest) {
        console.warn(`[AvatarPoser] _poseSpine: Rest not found: ${name}`);
      }
      if (bone && rest) {
        bone.rotation.z = rest.z - bodyTilt * 0.4;
        console.log(`[AvatarPoser] _poseSpine: Updated bone ${name}`);
      }
    });
  }

  // ─── Head / Neck ────────────────────────────────────────────────

  _poseHead(nose, leftShoulder, rightShoulder) {
    if (!nose || !leftShoulder || !rightShoulder) return;

    const neckX = (leftShoulder.x + rightShoulder.x) / 2;
    const neckY = (leftShoulder.y + rightShoulder.y) / 2;

    // Tilt: nose offset from neck centre
    const dx = nose.x - neckX;
    const dy = -(nose.y - neckY); // flip Y

    // Head lean left/right
    const headTiltZ = Math.atan2(dx, Math.abs(dy) + 0.05);

    // Head nod up/down (positive dy = nose above neck = looking level or up)
    const headNodX = Math.atan2(-dy, 0.25);

    const neckBones = ['neck1', 'neck2', 'neck3'];
    neckBones.forEach(name => {
      const bone = this.getBone(name);
      const rest = this.getRest(name);
      if (bone && rest) {
        bone.rotation.x = rest.x + headNodX * 0.12;
        bone.rotation.z = rest.z - headTiltZ * 0.12;
      }
    });

    const headBone = this.getBone('head');
    const headRest = this.getRest('head');
    if (headBone && headRest) {
      headBone.rotation.x = headRest.x + headNodX * 0.2;
      headBone.rotation.z = headRest.z - headTiltZ * 0.2;
    }
  }

  // ─── Arms (quaternion-based) ────────────────────────────────────

  _poseArmQ(side, shoulder, elbow, wrist, handLandmarks) {
    if (!shoulder || !elbow || !wrist) {
      console.warn(`[AvatarPoser] _poseArmQ(${side}): Missing joints`, { shoulder, elbow, wrist });
      return;
    }

    const upperArmName = this.BONE_NAMES[side === 'L' ? 'upperArmL' : 'upperArmR'];
    const forearmName  = this.BONE_NAMES[side === 'L' ? 'forearmL'  : 'forearmR'];
    const handName     = this.BONE_NAMES[side === 'L' ? 'handL'     : 'handR'];

    // Upper arm: shoulder → elbow direction
    const upperDir = this._landmarkDir(shoulder, elbow);
    if (upperDir) {
      this._poseBoneToDirection(upperArmName, upperDir);
      console.log(`[AvatarPoser] _poseArmQ(${side}): Updated upper arm ${upperArmName}`);
    } else {
      console.warn(`[AvatarPoser] _poseArmQ(${side}): No upper arm direction`);
    }

    // Forearm: elbow → wrist direction
    const foreDir = this._landmarkDir(elbow, wrist);
    if (foreDir) {
      this._poseBoneToDirection(forearmName, foreDir);
      console.log(`[AvatarPoser] _poseArmQ(${side}): Updated forearm ${forearmName}`);
    } else {
      console.warn(`[AvatarPoser] _poseArmQ(${side}): No forearm direction`);
    }

    // Hand: use hand landmarks wrist→middle-MCP if available, else follow forearm
    if (handLandmarks && handLandmarks.length >= 10) {
      const wristLm = handLandmarks[0];
      const middleMcp = handLandmarks[9];
      const handDir = this._landmarkDir(wristLm, middleMcp);
      if (handDir) {
        this._poseBoneToDirection(handName, handDir, 0.7);
        console.log(`[AvatarPoser] _poseArmQ(${side}): Updated hand ${handName} (hand landmarks)`);
      } else {
        console.warn(`[AvatarPoser] _poseArmQ(${side}): No hand direction from hand landmarks`);
      }
    } else if (foreDir) {
      this._poseBoneToDirection(handName, foreDir, 0.3);
      console.log(`[AvatarPoser] _poseArmQ(${side}): Updated hand ${handName} (forearm direction)`);
    } else {
      console.warn(`[AvatarPoser] _poseArmQ(${side}): No hand direction`);
    }
  }

  // ─── Fingers (euler curl) ───────────────────────────────────────

  _poseFingers(side, handLandmarks) {
    if (!handLandmarks || handLandmarks.length < 21) return;

    const suffix = side;
    const fingers = [
      { boneKeys: [`thumb${suffix}1`,  `thumb${suffix}2`,  `thumb${suffix}3`],  landmarks: [1, 2, 3, 4] },
      { boneKeys: [`index${suffix}1`,  `index${suffix}2`,  `index${suffix}3`],  landmarks: [5, 6, 7, 8] },
      { boneKeys: [`middle${suffix}1`, `middle${suffix}2`, `middle${suffix}3`], landmarks: [9, 10, 11, 12] },
      { boneKeys: [`ring${suffix}1`,   `ring${suffix}2`,   `ring${suffix}3`],   landmarks: [13, 14, 15, 16] },
      { boneKeys: [`pinky${suffix}1`,  `pinky${suffix}2`,  `pinky${suffix}3`],  landmarks: [17, 18, 19, 20] },
    ];

    fingers.forEach(finger => {
      const lm = finger.landmarks.map(i => handLandmarks[i]);
      if (!lm[0] || !lm[1] || !lm[2] || !lm[3]) return;

      for (let i = 0; i < 3; i++) {
        const bone = this.getBone(finger.boneKeys[i]);
        const rest = this.getRest(finger.boneKeys[i]);
        if (!bone || !rest) continue;

        if (i < 2) {
          // Compute bend angle between consecutive segments
          const seg1x = lm[i + 1].x - lm[i].x;
          const seg1y = -(lm[i + 1].y - lm[i].y);
          const seg2x = lm[i + 2].x - lm[i + 1].x;
          const seg2y = -(lm[i + 2].y - lm[i + 1].y);
          const cross = seg1x * seg2y - seg1y * seg2x;
          const dot   = seg1x * seg2x + seg1y * seg2y;
          const bend  = Math.atan2(cross, dot);
          bone.rotation.z = rest.z + bend * 0.6;
        } else {
          // Last segment: use absolute angle as fallback
          const segDir = {
            x: lm[i + 1].x - lm[i].x,
            y: -(lm[i + 1].y - lm[i].y)
          };
          const curl = Math.atan2(segDir.y, segDir.x);
          bone.rotation.z = rest.z + curl * 0.3;
        }
      }
    });
  }

  // ─── 2D skeleton fallback ───────────────────────────────────────

  draw2DSkeleton(landmarks) {
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    const drawPoint = (point, color, radius = 6) => {
      if (!point) return;
      ctx.beginPath();
      ctx.arc(point.x * w, point.y * h, radius, 0, 2 * Math.PI);
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

    if (landmarks.poseLandmarks) {
      ctx.shadowBlur = 12;
      ctx.shadowColor = 'rgba(167, 139, 250, 0.6)';
      this.POSE_CONNECTIONS.forEach(([start, end]) => {
        const s = landmarks.poseLandmarks[start];
        const e = landmarks.poseLandmarks[end];
        if (s && e) drawLine(s, e, 'rgba(167, 139, 250, 0.9)');
      });
      landmarks.poseLandmarks.forEach(lm => drawPoint(lm, 'rgba(167, 139, 250, 1)'));
    }

    if (landmarks.faceLandmarks) {
      landmarks.faceLandmarks.forEach((lm, i) => {
        if (i % 10 === 0) drawPoint(lm, 'rgba(255, 140, 66, 0.8)', 3);
      });
    }

    const drawHand = (handLandmarks, color) => {
      if (!handLandmarks || handLandmarks.length === 0) return;
      this.HAND_CONNECTIONS.forEach(([start, end]) => {
        const s = handLandmarks[start];
        const e = handLandmarks[end];
        if (s && e) drawLine(s, e, color);
      });
      handLandmarks.forEach(lm => drawPoint(lm, color, 5));
    };

    if (landmarks.rightHandLandmarks) drawHand(landmarks.rightHandLandmarks, 'rgba(76, 201, 240, 0.95)');
    if (landmarks.leftHandLandmarks)  drawHand(landmarks.leftHandLandmarks,  'rgba(6, 214, 160, 0.95)');

    ctx.shadowBlur = 0;
  }

  // ─── Resize / dispose ───────────────────────────────────────────

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
    this.restQuaternions = {};
    this.boneWorldRestQuaternions = {};
    this.boneRestDirections = {};

    if (this.canvas && this.container.contains(this.canvas)) {
      this.container.removeChild(this.canvas);
    }

    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this._isPosing = false;

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
