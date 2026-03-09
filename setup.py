#!/usr/bin/env python3
"""Setup script for Interpretable NLP Models."""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    commands = [
        ("pip install pre-commit", "Installing pre-commit"),
        ("pre-commit install", "Installing pre-commit hooks"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"⚠️  {description} failed, but continuing...")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "assets", 
        "data/raw",
        "data/processed",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True


def run_tests():
    """Run basic tests."""
    return run_command("python -m pytest tests/test_basic.py -v", "Running basic tests")


def main():
    """Main setup function."""
    print("🚀 Setting up Interpretable NLP Models")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Setup pre-commit
    print("\n🔧 Setting up pre-commit hooks...")
    setup_pre_commit()
    
    # Run tests
    print("\n🧪 Running tests...")
    if not run_tests():
        print("⚠️  Tests failed, but setup completed")
    
    print("\n✅ Setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Run the simple example: python examples/simple_example.py")
    print("2. Launch the demo: streamlit run demo/app.py")
    print("3. Train a model: python scripts/train.py --use_synthetic")
    print("4. Read the README.md for more information")
    
    print("\n⚠️  Remember to read the DISCLAIMER.md before using this tool!")


if __name__ == "__main__":
    main()
