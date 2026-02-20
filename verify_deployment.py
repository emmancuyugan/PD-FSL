#!/usr/bin/env python3
"""
Pre-deployment verification script for FSL Jetson optimization
Verifies all required components are in place and working
"""

import os
import sys
import subprocess
from pathlib import Path

class VerificationResults:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.warnings = []
    
    def add_pass(self, test_name, details=""):
        self.passed.append((test_name, details))
        print(f"✓ {test_name}")
        if details:
            print(f"  → {details}")
    
    def add_fail(self, test_name, details=""):
        self.failed.append((test_name, details))
        print(f"✗ {test_name}")
        if details:
            print(f"  → {details}")
    
    def add_warn(self, test_name, details=""):
        self.warnings.append((test_name, details))
        print(f"⚠ {test_name}")
        if details:
            print(f"  → {details}")
    
    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.warnings)
        print(f"\n{'='*50}")
        print(f"Verification Summary ({len(self.passed)}/{total} passed)")
        print(f"{'='*50}")
        if self.failed:
            print(f"\n❌ FAILED CHECKS ({len(self.failed)}):")
            for name, details in self.failed:
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for name, details in self.warnings:
                print(f"  - {name}")
                if details:
                    print(f"    {details}")
        print(f"\n✓  PASSED ({len(self.passed)})")
        return len(self.failed) == 0

def check_file_exists(path, description=""):
    """Check if file exists"""
    if os.path.exists(path):
        return True, f"{description or path} exists"
    return False, f"{description or path} missing"

def check_python_package(package_name, import_name=None):
    """Check if Python package is installed"""
    import_name = import_name or package_name
    try:
        __import__(import_name)
        return True, f"Package '{package_name}' installed"
    except ImportError:
        return False, f"Package '{package_name}' NOT installed"

def check_file_contains(filepath, search_text, description=""):
    """Check if file contains specific text"""
    if not os.path.exists(filepath):
        return False, f"{filepath} not found"
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            if search_text in content:
                return True, description or f"Found '{search_text}' in {filepath}"
            return False, f"'{search_text}' not found in {filepath}"
    except Exception as e:
        return False, str(e)

def run_python_check(code, description=""):
    """Run Python code and check if it succeeds"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, description or result.stdout.strip()
        return False, f"Exit code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    print("="*50)
    print("FSL Jetson Deployment Verification")
    print("="*50)
    
    results = VerificationResults()
    workspace = Path(__file__).parent
    
    # ======================================
    # 1. Core Application Files
    # ======================================
    print("\n[1] Core Application Files")
    print("-" * 50)
    
    files_to_check = [
        ("app.py", "Main Flask application"),
        ("model.py", "LSTM model definition"),
        ("jetson_optimization.py", "Jetson optimization module"),
        ("pathutils.py", "Path utilities"),
        ("init_db.py", "Database initializer"),
        ("requirements.txt", "Python dependencies"),
        ("run35.pth", "Model weights"),
    ]
    
    for filename, description in files_to_check:
        exists, msg = check_file_exists(workspace / filename, description)
        if exists:
            results.add_pass(f"{filename}", msg)
        else:
            results.add_fail(f"{filename}", msg)
    
    # ======================================
    # 2. Template Files
    # ======================================
    print("\n[2] HTML Templates")
    print("-" * 50)
    
    templates = [
        ("templates/select.html", "SELECT mode"),
        ("templates/activity.html", "ACTIVITY mode"),
        ("templates/auto.html", "AUTO mode"),
        ("templates/detect.html", "DETECT mode"),
        ("templates/results.html", "Results dashboard"),
        ("templates/base.html", "Base template"),
    ]
    
    for filename, description in templates:
        exists, msg = check_file_exists(workspace / filename, description)
        if exists:
            results.add_pass(f"{description}", msg)
        else:
            results.add_fail(f"{description}", msg)
    
    # ======================================
    # 3. Optimization Code Presence
    # ======================================
    print("\n[3] Optimization Code Integration")
    print("-" * 50)
    
    # Check app.py for JETSON_ENABLED
    exists, msg = check_file_contains(
        workspace / "app.py",
        "JETSON_ENABLED",
        "JETSON_ENABLED environment variable setup"
    )
    if exists:
        results.add_pass("JETSON_ENABLED flag", msg)
    else:
        results.add_warn("JETSON_ENABLED flag", msg)
    
    # Check for mixed precision in app.py
    exists, msg = check_file_contains(
        workspace / "app.py",
        "torch.cuda.amp.autocast",
        "Mixed precision (autocast) context"
    )
    if exists:
        results.add_pass("Mixed precision context", msg)
    else:
        results.add_fail("Mixed precision context", msg)
    
    # Check for inference_mode
    exists, msg = check_file_contains(
        workspace / "app.py",
        "torch.inference_mode",
        "Inference mode optimization"
    )
    if exists:
        results.add_pass("Inference mode", msg)
    else:
        results.add_warn("Inference mode", msg)
    
    # Check for jetson_optimization import
    exists, msg = check_file_contains(
        workspace / "app.py",
        "from jetson_optimization import",
        "Jetson optimization module imported"
    )
    if exists:
        results.add_pass("Jetson optimization import", msg)
    else:
        results.add_warn("Jetson optimization import", msg)
    
    # ======================================
    # 4. Source Parameter in Templates
    # ======================================
    print("\n[4] Source Parameter Tracking")
    print("-" * 50)
    
    # Check select.html for source: 'select'
    exists, msg = check_file_contains(
        workspace / "templates/select.html",
        "source: 'select'",
        "SELECT mode source parameter"
    )
    if exists:
        results.add_pass("SELECT mode source tracking", msg)
    else:
        results.add_fail("SELECT mode source tracking", msg)
    
    # Check activity.html for source: 'activity'
    exists, msg = check_file_contains(
        workspace / "templates/activity.html",
        "source: 'activity'",
        "ACTIVITY mode source parameter"
    )
    if exists:
        results.add_pass("ACTIVITY mode source tracking", msg)
    else:
        results.add_fail("ACTIVITY mode source tracking", msg)
    
    # Check auto.html for source: 'auto' OR /predict_auto (AUTO mode saves source internally)
    exists, msg = check_file_contains(
        workspace / "templates/auto.html",
        "/predict_auto",
        "AUTO mode endpoint"
    )
    if exists:
        results.add_pass("AUTO mode source tracking", "AUTO mode uses /predict_auto endpoint (source handled internally in backend)")
    else:
        results.add_fail("AUTO mode source tracking", msg)
    
    # ======================================
    # 5. Frontend Authentication
    # ======================================
    print("\n[5] Frontend Session Authentication")
    print("-" * 50)
    
    # Check for credentials: 'include'
    exists, msg = check_file_contains(
        workspace / "templates/results.html",
        "credentials: 'include'",
        "Session cookie forwarding in fetch"
    )
    if exists:
        results.add_pass("Session credentials in fetch", msg)
    else:
        results.add_fail("Session credentials in fetch", msg)
    
    # Check for /api/results endpoint
    exists, msg = check_file_contains(
        workspace / "templates/results.html",
        "/api/results",
        "Results API endpoint"
    )
    if exists:
        results.add_pass("Results API endpoint", msg)
    else:
        results.add_fail("Results API endpoint", msg)
    
    # ======================================
    # 6. Python Dependencies
    # ======================================
    print("\n[6] Python Package Installation")
    print("-" * 50)
    
    packages = [
        ("torch", "PyTorch", "torch"),
        ("flask", "Flask", "flask"),
        ("sqlalchemy", "SQLAlchemy", "sqlalchemy"),
        ("numpy", "NumPy", "numpy"),
    ]
    
    for package_name, description, import_name in packages:
        exists, msg = check_python_package(description, import_name)
        if exists:
            results.add_pass(f"{description} installed", msg)
        else:
            results.add_warn(f"{description} not installed", msg)
    
    # ======================================
    # 7. Python Syntax Validation
    # ======================================
    print("\n[7] Python Syntax Validation")
    print("-" * 50)
    
    python_files = [
        (workspace / "app.py", "app.py"),
        (workspace / "model.py", "model.py"),
        (workspace / "jetson_optimization.py", "jetson_optimization.py"),
    ]
    
    for filepath, filename in python_files:
        if filepath.exists():
            exists, msg = run_python_check(
                f"import py_compile; py_compile.compile(r'{filepath}', doraise=True)",
                f"{filename} syntax valid"
            )
            if exists:
                results.add_pass(f"{filename} syntax", msg)
            else:
                results.add_fail(f"{filename} syntax", msg)
    
    # ======================================
    # 8. PyTorch Configuration
    # ======================================
    print("\n[8] PyTorch Configuration")
    print("-" * 50)
    
    # Check PyTorch version
    exists, msg = run_python_check(
        "import torch; print(f'PyTorch {torch.__version__}')",
        "PyTorch version"
    )
    if exists:
        results.add_pass("PyTorch version", msg)
    else:
        results.add_warn("PyTorch version", msg)
    
    # Check CUDA availability (may be unavailable on dev PC)
    exists, msg = run_python_check(
        "import torch; print(f'CUDA available: {torch.cuda.is_available()}')",
        "CUDA availability check"
    )
    if exists:
        if "True" in msg or "CUDA available: True" in msg:
            results.add_pass("CUDA support", msg)
        else:
            results.add_warn("CUDA not available (normal on CPU-only dev PC)", msg)
    
    # ======================================
    # 9. Deployment Documentation
    # ======================================
    print("\n[9] Deployment Documentation")
    print("-" * 50)
    
    docs = [
        ("JETSON_OPTIMIZATION.md", "Jetson optimization guide"),
        ("DEPLOYMENT_CHECKLIST.md", "Deployment checklist"),
        ("PERFORMANCE_BASELINE.md", "Performance baseline report"),
        ("deploy_jetson.sh", "Automated deployment script"),
    ]
    
    for filename, description in docs:
        exists, msg = check_file_exists(workspace / filename, description)
        if exists:
            results.add_pass(f"{description}", msg)
        else:
            results.add_warn(f"{description}", msg)
    
    # ======================================
    # Summary
    # ======================================
    success = results.summary()
    
    if success:
        print("\n✅ All critical checks passed! Ready for Jetson deployment.")
        print("\nNext steps:")
        print("1. Review JETSON_OPTIMIZATION.md for optimization details")
        print("2. Check PERFORMANCE_BASELINE.md for expected performance")
        print("3. Run deploy_jetson.sh to deploy to Jetson hardware")
        print("4. Follow DEPLOYMENT_CHECKLIST.md for post-deployment validation")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix above issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
