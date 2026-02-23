#!/bin/bash
# Pre-deployment verification script for Jetson Orin Nano
# Run this BEFORE building the C++ module

set -e

echo "=================================================="
echo "🔍 PD-FSL C++ Camera Handler - Pre-Deployment Check"
echo "=================================================="

FAILED=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found ($(which $1))"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT FOUND"
        return 1
    fi
}

check_package() {
    if pkg-config --exists $1 2>/dev/null; then
        VERSION=$(pkg-config --modversion $1)
        echo -e "${GREEN}✓${NC} $1 installed (version $VERSION)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 NOT FOUND"
        return 1
    fi
}

check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Python module '$1' available"
        return 0
    else
        echo -e "${RED}✗${NC} Python module '$1' NOT FOUND"
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} File exists: $1"
        return 0
    else
        echo -e "${YELLOW}⊙${NC} File not found: $1"
        return 1
    fi
}

# ====================================
echo -e "\n${YELLOW}[1/5] System Tools${NC}"
# ====================================
check_command "python3" || ((FAILED++))
check_command "g++" || ((FAILED++))
check_command "cmake" || ((FAILED++))
check_command "protoc" || ((FAILED++))
check_command "ls" || ((FAILED++))

# ====================================
echo -e "\n${YELLOW}[2/5] System Libraries${NC}"
# ====================================
check_package "opencv4" || check_package "opencv" || ((WARNINGS++))
check_package "protobuf" || ((FAILED++))
check_package "libssl" || ((WARNINGS++))

# Verify OpenCV components
if pkg-config --exists opencv4; then
    echo -e "  Checking OpenCV components..."
    pkg-config --cflags opencv4 | grep -q "I/usr" && echo -e "  ${GREEN}✓${NC} OpenCV includes found" || ((WARNINGS++))
fi

# ====================================
echo -e "\n${YELLOW}[3/5] Python Modules${NC}"
# ====================================
check_python_module "numpy" || ((FAILED++))
check_python_module "mediapipe" || ((FAILED++))
check_python_module "cv2" || ((FAILED++))
check_python_module "PIL" || ((WARNINGS++))

# ====================================
echo -e "\n${YELLOW}[4/5] Hardware & Devices${NC}"
# ====================================
if [ -e /dev/video0 ]; then
    echo -e "${GREEN}✓${NC} Camera device found: /dev/video0"
else
    echo -e "${RED}✗${NC} Camera device /dev/video0 NOT FOUND"
    echo "  Available video devices:"
    ls /dev/video* 2>/dev/null || echo "    (none)"
    ((FAILED++))
fi

# ====================================
echo -e "\n${YELLOW}[5/5] Jetson-Specific${NC}"
# ====================================
if [ -f /etc/nv_tegra_release ]; then
    JETSON_MODEL=$(grep "JETSON_BOARD" /etc/nv_tegra_release | cut -d'=' -f2)
    echo -e "${GREEN}✓${NC} Jetson platform detected: $JETSON_MODEL"
    
    # Check for ARMv8 (aarch64)
    if [ "$(uname -m)" = "aarch64" ]; then
        echo -e "${GREEN}✓${NC} Architecture is aarch64 (ARM64)"
    else
        echo -e "${YELLOW}⊙${NC} Warning: Architecture is $(uname -m), expected aarch64"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⊙${NC} Jetson device not detected (might be running on desktop Linux)"
fi

# ====================================
echo -e "\n${YELLOW}[MEDIAPIPE CHECK]${NC}"
# ====================================
MEDIAPIPE_PATH=$(python3 -c "import mediapipe; import os; print(os.path.dirname(mediapipe.__file__))" 2>/dev/null)
if [ -n "$MEDIAPIPE_PATH" ]; then
    echo -e "${GREEN}✓${NC} Mediapipe found at: $MEDIAPIPE_PATH"
    
    # Check for holistic
    if [ -f "$MEDIAPIPE_PATH/tasks/python/vision/holistic_landmarker.py" ] || \
       [ -d "$MEDIAPIPE_PATH/solutions" ]; then
        echo -e "${GREEN}✓${NC} Mediapipe Holistic solution available"
    else
        echo -e "${YELLOW}⊙${NC} Mediapipe Holistic not found in standard location"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} Mediapipe not found"
    ((FAILED++))
fi

# ====================================
echo -e "\n${YELLOW}[PROTOBUF CHECK]${NC}"
# ====================================
PROTOBUF_VERSION=$(protoc --version 2>/dev/null | awk '{print $NF}')
if [ -n "$PROTOBUF_VERSION" ]; then
    echo -e "${GREEN}✓${NC} Protobuf version: $PROTOBUF_VERSION"
    if [ -d "/usr/include/google/protobuf" ]; then
        echo -e "${GREEN}✓${NC} Protobuf development headers found"
    else
        echo -e "${YELLOW}⊙${NC} Protobuf headers not in standard location"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} Protobuf not found"
    ((FAILED++))
fi

# ====================================
echo -e "\n=================================================="
echo "📊 SUMMARY"
echo "=================================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ No warnings${NC}"
        echo ""
        echo "📝 Ready to proceed with build. Run:"
        echo "   mkdir -p build && cd build"
        echo "   cmake -DMEDIAPIPE_DIR=/usr/local/mediapipe .."
        echo "   cmake --build . -j\$(nproc)"
        exit 0
    else
        echo -e "${YELLOW}⊙ $WARNINGS warning(s) found (non-critical)${NC}"
        echo ""
        echo "You can proceed, but may need to adjust paths during cmake."
        exit 0
    fi
else
    echo -e "${RED}✗ $FAILED critical check(s) FAILED${NC}"
    echo ""
    echo "❌ Installation incomplete. Fix issues above before building."
    echo ""
    echo "Common fixes:"
    echo "  • sudo apt update && sudo apt install build-essential cmake"
    echo "  • sudo apt install libopencv-dev libprotobuf-dev protobuf-compiler"
    echo "  • pip3 install mediapipe opencv-python numpy"
    exit 1
fi
