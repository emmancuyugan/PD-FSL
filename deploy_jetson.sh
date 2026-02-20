#!/bin/bash
# Jetson Orin Nano Super Deployment Script
# Usage: bash deploy_jetson.sh <jetson_ip> [jetson_username]

set -e

JETSON_IP=${1:-localhost}
JETSON_USER=${2:-jetson}
JETSON_HOME=/home/$JETSON_USER

echo "=========================================="
echo "FSL Jetson Orin Nano Super Deployment"
echo "=========================================="
echo "Target: $JETSON_USER@$JETSON_IP"
echo ""

# Check if Jetson is reachable
echo "[1/6] Verifying Jetson connectivity..."
if ! ping -c 1 "$JETSON_IP" &> /dev/null; then
    echo "ERROR: Cannot reach $JETSON_IP"
    exit 1
fi
echo "✓ Jetson is reachable"

# Create deployment directory on Jetson
echo "[2/6] Setting up deployment directory..."
ssh "$JETSON_USER@$JETSON_IP" "mkdir -p $JETSON_HOME/fsl/{templates,static,utils,__pycache__}"
echo "✓ Directories created"

# Copy Python application files
echo "[3/6] Transferring application files..."
scp app.py model.py jetson_optimization.py pathutils.py init_db.py requirements.txt \
    "$JETSON_USER@$JETSON_IP:$JETSON_HOME/fsl/"

# Copy HTML templates
echo "    Transferring templates..."
for template in templates/*.html; do
    scp "$template" "$JETSON_USER@$JETSON_IP:$JETSON_HOME/fsl/templates/"
done

# Copy static files (CSS, JS)
echo "    Transferring static assets..."
scp -r static/* "$JETSON_USER@$JETSON_IP:$JETSON_HOME/fsl/static/"

echo "✓ Files transferred"

# Copy model weights
echo "[4/6] Transferring model weights..."
if [ -f "run35.pth" ]; then
    scp run35.pth "$JETSON_USER@$JETSON_IP:$JETSON_HOME/fsl/"
    echo "✓ Model weights transferred"
else
    echo "⚠ run35.pth not found - model must be present on Jetson"
fi

# Install dependencies on Jetson
echo "[5/6] Installing Python dependencies..."
ssh "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/fsl
echo "Installing pip packages..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install PyTorch (CUDA 12.x for JetPack 5.1+)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1

# Install other dependencies
pip install flask flask-cors flask-sqlalchemy python-dotenv numpy opencv-python > /dev/null 2>&1

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
EOF
echo "✓ Dependencies installed"

# Enable Jetson optimizations and start service
echo "[6/6] Configuring services..."
ssh "$JETSON_USER@$JETSON_IP" << 'EOF'
cd ~/fsl

# Enable Jetson optimizations
export JETSON_OPTIMIZED=true
export FLASK_APP=app.py
export FLASK_ENV=production

# Initialize database if needed
python3 init_db.py > /dev/null 2>&1 || true

# Create systemd service for auto-start
echo "Creating systemd service..."
sudo bash -c 'cat > /etc/systemd/system/fsl-jetson.service << "EOFSERVICE"
[Unit]
Description=FSL Detection Service - Jetson Orin Nano
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/fsl
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="JETSON_OPTIMIZED=true"
ExecStart=/usr/bin/python3 /home/jetson/fsl/app.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOFSERVICE'

sudo systemctl daemon-reload
sudo systemctl enable fsl-jetson.service
echo "✓ Service installed and enabled"
EOF

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH into Jetson: ssh $JETSON_USER@$JETSON_IP"
echo "2. Test the service: sudo systemctl start fsl-jetson"
echo "3. Check status: sudo systemctl status fsl-jetson"
echo "4. View logs: journalctl -u fsl-jetson -f"
echo "5. Test API: curl http://localhost:5000/health"
echo ""
echo "To monitor performance on Jetson:"
echo "  watch nvidia-smi          # GPU monitoring"
echo "  top -p \$(pgrep python)    # CPU monitoring"
echo ""
