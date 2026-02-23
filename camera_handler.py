"""
Python wrapper for C++ Camera Handler with Mediapipe
Enables fast camera capture and landmark processing in C++
"""

import ctypes
import os
import struct
import numpy as np
from typing import Optional, Tuple, List
import base64
import io
from PIL import Image
import threading
import time

# ============================================
# Constants
# ============================================
MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB for frame + landmarks
KEYPOINT_SIZE = 16  # 4 floats (x, y, z, visibility)

# ============================================
# Data structures
# ============================================
class KeyPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float),
                ("visibility", ctypes.c_float)]


# ============================================
# Camera Handler Python Wrapper
# ============================================
class CameraCPPHandler:
    """Python interface to C++ camera handler with Mediapipe"""
    
    def __init__(self, lib_path: Optional[str] = None, device_id: int = 0):
        """
        Initialize the camera handler
        
        Args:
            lib_path: Path to compiled camera_handler shared library (.so or .dll)
            device_id: Camera device ID (default 0)
        """
        self.lib_path = lib_path
        self.device_id = device_id
        self.lib = None
        self.is_running = False
        self.frame_buffer = None
        self._load_library()
        
    def _load_library(self):
        """Load the compiled C++ shared library"""
        if self.lib_path is None:
            # Try to find library in common locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "build", "libcamera_handler.so"),
                os.path.join(os.path.dirname(__file__), "build", "camera_handler.dll"),
                "./build/libcamera_handler.so",
                "./build/camera_handler.dll",
                "/usr/local/lib/libcamera_handler.so",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.lib_path = path
                    break
            
            if self.lib_path is None:
                raise RuntimeError(
                    f"Could not find camera_handler library. "
                    f"Please specify lib_path or build the C++ module:\n"
                    f"  mkdir build && cd build\n"
                    f"  cmake -DMEDIAPIPE_DIR=/path/to/mediapipe ..\n"
                    f"  cmake --build ."
                )
        
        print(f"Loading camera handler library: {self.lib_path}")
        self.lib = ctypes.CDLL(self.lib_path)
        
        # Define function signatures
        self.lib.camera_init.argtypes = [ctypes.c_int]
        self.lib.camera_init.restype = ctypes.c_int
        
        self.lib.camera_start.argtypes = []
        self.lib.camera_start.restype = ctypes.c_int
        
        self.lib.camera_stop.argtypes = []
        self.lib.camera_stop.restype = None
        
        self.lib.camera_get_frame.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.camera_get_frame.restype = ctypes.c_int
        
        self.lib.camera_cleanup.argtypes = []
        self.lib.camera_cleanup.restype = None
    
    def initialize(self) -> bool:
        """Initialize camera and Mediapipe"""
        if not self.lib:
            print("ERROR: Library not loaded")
            return False
        
        result = self.lib.camera_init(self.device_id)
        if result == 0:
            print("ERROR: Failed to initialize camera or Mediapipe")
            return False
        
        print("Camera handler initialized successfully")
        return True
    
    def start(self) -> bool:
        """Start camera capture and processing"""
        if not self.lib:
            return False
        
        result = self.lib.camera_start()
        if result == 0:
            print("ERROR: Failed to start camera processing")
            return False
        
        self.is_running = True
        print("Camera processing started")
        return True
    
    def stop(self):
        """Stop camera capture and processing"""
        if self.lib and self.is_running:
            self.lib.camera_stop()
            self.is_running = False
            print("Camera processing stopped")
    
    def get_frame(self) -> Optional[dict]:
        """
        Get the latest processed frame with landmarks
        
        Returns:
            Dictionary with:
            - 'pose': list of (x, y, z) tuples
            - 'left_hand': list of (x, y, z) tuples
            - 'right_hand': list of (x, y, z) tuples
            - 'frame_bytes': JPEG-encoded frame
            - 'frame_b64': Base64 encoded frame (for web display)
            - 'timestamp': frame timestamp
        """
        if not self.lib or not self.is_running:
            return None
        
        # Allocate buffer
        buffer = ctypes.create_string_buffer(MAX_BUFFER_SIZE)
        pose_count = ctypes.c_int()
        left_hand_count = ctypes.c_int()
        right_hand_count = ctypes.c_int()
        
        # Call C++ function
        bytes_read = self.lib.camera_get_frame(
            buffer,
            MAX_BUFFER_SIZE,
            ctypes.byref(pose_count),
            ctypes.byref(left_hand_count),
            ctypes.byref(right_hand_count)
        )
        
        if bytes_read <= 0:
            return None  # No frame available
        
        # Parse binary data
        offset = 0
        header_size = 4 * ctypes.sizeof(ctypes.c_int)
        
        # Read counts from header
        pose_count_val = pose_count.value
        left_hand_count_val = left_hand_count.value
        right_hand_count_val = right_hand_count.value
        
        # Parse landmarks from struct format: 16 bytes per keypoint (4 floats)
        offset = header_size
        
        pose_landmarks = []
        for i in range(pose_count_val):
            kp = KeyPoint.from_buffer_copy(buffer.raw[offset:offset + KEYPOINT_SIZE])
            pose_landmarks.append((kp.x, kp.y, kp.z))
            offset += KEYPOINT_SIZE
        
        left_hand_landmarks = []
        for i in range(left_hand_count_val):
            kp = KeyPoint.from_buffer_copy(buffer.raw[offset:offset + KEYPOINT_SIZE])
            left_hand_landmarks.append((kp.x, kp.y, kp.z))
            offset += KEYPOINT_SIZE
        
        right_hand_landmarks = []
        for i in range(right_hand_count_val):
            kp = KeyPoint.from_buffer_copy(buffer.raw[offset:offset + KEYPOINT_SIZE])
            right_hand_landmarks.append((kp.x, kp.y, kp.z))
            offset += KEYPOINT_SIZE
        
        # Remaining bytes are JPEG frame data
        frame_size = bytes_read - offset
        frame_bytes = buffer.raw[offset:offset + frame_size]
        
        # Encode to base64 for web
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')
        
        return {
            'pose': pose_landmarks,
            'left_hand': left_hand_landmarks,
            'right_hand': right_hand_landmarks,
            'frame_bytes': frame_bytes,
            'frame_b64': frame_b64,
            'timestamp': time.time()
        }
    
    def get_landmarks_only(self) -> Optional[List[float]]:
        """
        Get only the landmarks as a flat float array for model inference
        Compatible with the JavaScript preprocessing
        """
        frame_data = self.get_frame()
        if not frame_data:
            return None
        
        # Replicate the JavaScript feature extraction
        landmarks = []
        
        # Add all pose and hand landmarks
        landmarks.extend([coord for point in frame_data['pose'] for coord in point])
        landmarks.extend([coord for point in frame_data['left_hand'] for coord in point])
        landmarks.extend([coord for point in frame_data['right_hand'] for coord in point])
        
        return landmarks
    
    def cleanup(self):
        """Clean up resources"""
        if self.lib:
            self.stop()
            self.lib.camera_cleanup()
            print("Camera handler cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# ============================================
# Convenience functions
# ============================================
_global_handler = None

def init_camera(lib_path: Optional[str] = None, device_id: int = 0) -> CameraCPPHandler:
    """Initialize global camera handler"""
    global _global_handler
    _global_handler = CameraCPPHandler(lib_path, device_id)
    if not _global_handler.initialize():
        raise RuntimeError("Failed to initialize camera handler")
    return _global_handler

def start_camera() -> bool:
    """Start global camera handler"""
    global _global_handler
    if _global_handler is None:
        raise RuntimeError("Camera handler not initialized. Call init_camera() first")
    return _global_handler.start()

def get_camera_frame() -> Optional[dict]:
    """Get frame from global camera handler"""
    global _global_handler
    if _global_handler is None:
        return None
    return _global_handler.get_frame()

def stop_camera():
    """Stop global camera handler"""
    global _global_handler
    if _global_handler:
        _global_handler.stop()

def cleanup_camera():
    """Clean up global camera handler"""
    global _global_handler
    if _global_handler:
        _global_handler.cleanup()
        _global_handler = None


# ============================================
# Testing
# ============================================
if __name__ == "__main__":
    print("Starting camera handler test...")
    
    try:
        # Initialize
        handler = CameraCPPHandler()
        handler.initialize()
        handler.start()
        
        print("Capturing 10 frames...")
        for i in range(10):
            frame = handler.get_frame()
            if frame:
                print(f"Frame {i}: "
                      f"Pose={len(frame['pose'])} points, "
                      f"L_Hand={len(frame['left_hand'])} points, "
                      f"R_Hand={len(frame['right_hand'])} points, "
                      f"JPEG={len(frame['frame_bytes'])} bytes")
            else:
                print(f"Frame {i}: No data available")
            
            time.sleep(0.1)
        
        handler.stop()
        handler.cleanup()
        print("Test complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
