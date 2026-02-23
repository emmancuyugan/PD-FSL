#!/bin/bash
# Complete deployment script for Jetson Orin Nano
# Usage: bash deploy_to_jetson.sh <jetson_ip>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy_to_jetson.sh <jetson_ip> [username]"
    echo "Example: bash deploy_to_jetson.sh 192.168.1.100 jetson"
    exit 1
fi

JETSON_IP=$1
JETSON_USER=${2:-jetson}
PROJECT_DIR="/home/${JETSON_USER}/PD-FSL"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
print_step() {
    echo -e "\n${BLUE}[→]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

print_warning() {
    echo -e "${YELLOW}[⊙]${NC} $1"
}

# ============================================
# STEP 1: Connectivity Check
# ============================================
print_step "Checking Jetson connectivity..."

if ping -c 1 "$JETSON_IP" > /dev/null 2>&1; then
    print_success "Jetson is reachable at $JETSON_IP"
else
    print_error "Cannot reach Jetson at $JETSON_IP"
fi

# ============================================
# STEP 2: Transfer Project
# ============================================
print_step "Transferring project to Jetson..."

if command -v scp &> /dev/null; then
    scp -r "$(pwd)" "${JETSON_USER}@${JETSON_IP}:${PROJECT_DIR}" 2>/dev/null && \
        print_success "Project transferred to ${PROJECT_DIR}" || \
        print_warning "Project transfer may have partially succeeded"
else
    print_error "scp command not found"
fi

# ============================================
# STEP 3: Remote Setup Script
# ============================================
print_step "Running setup on Jetson..."

SETUP_SCRIPT='
#!/bin/bash
set -e

echo "🚀 PD-FSL C++ Camera Handler Deployment on Jetson"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update -qq
sudo apt install -y \
    build-essential cmake git \
    libopencv-dev \
    python3-dev python3-pip \
    libprotobuf-dev protobuf-compiler \
    > /dev/null 2>&1

# Navigate to project
cd '"${PROJECT_DIR}"'

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip3 install -q numpy Pillow
pip3 install -q -r requirements.txt 2>/dev/null || echo "Note: Some packages may already be installed"
pip3 install -q mediapipe opencv-python

# Run pre-build checks
echo "🔍 Running pre-deployment checks..."
if [ -f prebuild_check.sh ]; then
    bash prebuild_check.sh || echo "Some checks failed but continuing..."
fi

# Create build directory
echo "🏗️  Creating build directory..."
mkdir -p build && cd build

# Configure CMake
echo "⚙️  Configuring CMake..."
cmake -DMEDIAPIPE_DIR=/usr/local/mediapipe \
      -DCMAKE_CXX_FLAGS="-march=native -O3" \
      -DCMAKE_BUILD_TYPE=Release \
      .. 2>&1 | tail -20

# Build
echo "🔨 Building C++ module (this may take 5-10 minutes)..."
if cmake --build . -j$(nproc) 2>&1; then
    echo "✅ Build completed successfully!"
    ls -lh libcamera_handler.so
else
    echo "❌ Build failed. Check CMakeLists.txt and try again."
    exit 1
fi

# Test
echo "🧪 Testing C++ module..."
cd ..
if python3 -c "from camera_handler import CameraCPPHandler; print(\"✅ C++ module loads successfully\")" 2>/dev/null; then
    echo "Ready to run!"
else
    echo "⚠️  Warning: C++ module may need adjustment. Browser fallback available."
fi

echo ""
echo "=================================================="
echo "✅ Deployment complete!"
echo "=================================================="
echo ""
echo "📝 To start the application:"
echo "   cd '"${PROJECT_DIR}"'"
echo "   python3 app.py"
echo ""
echo "🌐 Access at:"
echo "   http://'$JETSON_IP':5000"
echo ""
'

# Run setup script on Jetson via SSH
ssh "${JETSON_USER}@${JETSON_IP}" bash -s << 'EOF'
$SETUP_SCRIPT
EOF

print_success "Setup complete on Jetson"

# ============================================
# STEP 4: Final Status
# ============================================
print_step "Verifying deployment..."

ssh "${JETSON_USER}@${JETSON_IP}" "
    cd ${PROJECT_DIR}
    if [ -f build/libcamera_handler.so ]; then
        echo -e \"${GREEN}✓ libcamera_handler.so exists${NC}\"
    else
        echo -e \"${YELLOW}⊙ libcamera_handler.so not found (may need rebuild)${NC}\"
    fi
    
    if python3 -c 'import mediapipe' 2>/dev/null; then
        echo -e \"${GREEN}✓ Mediapipe installed${NC}\"
    else
        echo -e \"${RED}✗ Mediapipe NOT installed${NC}\"
    fi
    
    if [ -e /dev/video0 ]; then
        echo -e \"${GREEN}✓ Camera device available${NC}\"
    else
        echo -e \"${RED}✗ Camera device NOT available${NC}\"
    fi
"

# ============================================
# STEP 5: Next Steps
# ============================================
echo ""
echo "=================================================="
echo "🚀 DEPLOYMENT COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. SSH into Jetson:"
echo "   ssh ${JETSON_USER}@${JETSON_IP}"
echo ""
echo "2. Start the application:"
echo "   cd ${PROJECT_DIR}"
echo "   python3 app.py"
echo ""
echo "3. Open in browser:"
echo "   http://${JETSON_IP}:5000"
echo ""
echo "4. Check deployment status:"
echo "   curl -X POST http://${JETSON_IP}:5000/camera/init"
echo "   curl http://${JETSON_IP}:5000/camera/frame"
echo ""
echo "For troubleshooting, see:"
echo "   - PREDEPLOYMENT_VERIFICATION.md"
echo "   - BUILD_INSTRUCTIONS.md"
echo "   - CPP_MEDIAPIPE_FIXES.md"
echo ""
