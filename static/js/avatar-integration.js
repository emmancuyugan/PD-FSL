/**
 * Avatar Integration Helper
 * 
 * This module integrates the AvatarPoser into SELECT and ACTIVITY modes
 * to replace the 2D skeleton with 3D avatars.
 * 
 * Usage:
 *   import { initAvatarFeedback, updateAvatar, disposeAvatar } from './avatar-integration.js';
 *   
 *   // Initialize when showing skeleton display
 *   const avatar = await initAvatarFeedback('skeletonCanvas', 'models/boy.glb');
 *   
 *   // Update with landmark data (call this every frame)
 *   updateAvatar(avatar, landmarkData);
 *   
 *   // Cleanup when closing
 *   disposeAvatar(avatar);
 */

// Model paths - configure these to point to your converted .glb files
export const MODEL_PATHS = {
  boy: 'static/models/boy.glb',
  girl: 'static/models/boy.glb', // Use boy as fallback until girl model is added
  fallback: 'static/models/boy.glb'
};

/**
 * Initialize the 3D avatar in the skeleton container
 * @param {string} canvasId - ID of the canvas element (e.g., 'skeletonCanvas')
 * @param {string} modelPath - Path to the .glb model file
 * @returns {Promise<AvatarPoser|null>} - Returns the AvatarPoser instance or null if failed
 */
export async function initAvatarFeedback(canvasId, modelPath = MODEL_PATHS.boy) {
  const container = document.getElementById(canvasId);
  if (!container) {
    console.error('[AvatarIntegration] Container not found:', canvasId);
    return null;
  }

  try {
    // Dynamically import the AvatarPoser (ES module)
    const { default: AvatarPoser } = await import('./vrm-poser.js');
    
    const avatar = new AvatarPoser(container, modelPath, {
      autoScale: true,
      autoCenter: true,
      smoothingFactor: 0.4
    });

    await avatar.init();
    console.log('[AvatarIntegration] Avatar initialized with model:', modelPath);
    
    return avatar;
  } catch (error) {
    console.error('[AvatarIntegration] Failed to initialize avatar:', error);
    return null;
  }
}

/**
 * Update the avatar pose with new landmark data
 * @param {AvatarPoser} avatar - The AvatarPoser instance
 * @param {Object} landmarks - MediaPipe landmark data
 */
export function updateAvatar(avatar, landmarks) {
  if (!avatar) return;
  
  // Convert server landmarks format to what AvatarPoser expects
  const formattedLandmarks = {
    poseLandmarks: landmarks.poseLandmarks || [],
    faceLandmarks: landmarks.faceLandmarks || [],
    leftHandLandmarks: landmarks.leftHandLandmarks || [],
    rightHandLandmarks: landmarks.rightHandLandmarks || []
  };
  
  avatar.updateLandmarks(formattedLandmarks);
}

/**
 * Dispose of the avatar instance
 * @param {AvatarPoser} avatar - The AvatarPoser instance to dispose
 */
export function disposeAvatar(avatar) {
  if (avatar) {
    avatar.dispose();
    console.log('[AvatarIntegration] Avatar disposed');
  }
}

/**
 * Switch between boy/girl models
 * @param {AvatarPoser} avatar - Current AvatarPoser instance
 * @param {string} gender - 'boy' or 'girl'
 * @returns {Promise<AvatarPoser>} - Returns new AvatarPoser instance
 */
export async function switchAvatarModel(avatar, gender) {
  // Dispose old avatar
  disposeAvatar(avatar);
  
  // Get new model path
  const modelPath = MODEL_PATHS[gender] || MODEL_PATHS.boy;
  
  // Initialize new avatar
  const container = document.getElementById('skeletonCanvas');
  return await initAvatarFeedback('skeletonCanvas', modelPath);
}

/**
 * Create a skeleton display element that uses 3D avatar
 * Call this to replace the existing skeleton canvas with 3D avatar
 */
export async function showAvatarFeedback(landmarkData, gender = 'boy') {
  // Get the skeleton display container
  const skeletonDisplay = document.getElementById('skeletonDisplay');
  if (!skeletonDisplay) {
    console.error('[AvatarIntegration] skeletonDisplay not found');
    return null;
  }

  // Clear existing content
  skeletonDisplay.innerHTML = '';
  
  // Create a new container for the 3D avatar
  const avatarContainer = document.createElement('div');
  avatarContainer.id = 'avatarContainer';
  avatarContainer.style.width = '100%';
  avatarContainer.style.height = '100%';
  skeletonDisplay.appendChild(avatarContainer);

  // Initialize the avatar
  const modelPath = MODEL_PATHS[gender] || MODEL_PATHS.boy;
  const avatar = await initAvatarFeedback('avatarContainer', modelPath);
  
  // If we have initial landmark data, update immediately
  if (landmarkData) {
    updateAvatar(avatar, landmarkData);
  }

  return avatar;
}

// Global cache for avatar instances (used by select.html and activity.html)
window.avatarCache = {
  select: null,
  activity: null
};