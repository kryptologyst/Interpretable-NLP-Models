#!/usr/bin/env python3
"""Quick start script for Interpretable NLP Models."""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Quick start demo."""
    print("🚀 Interpretable NLP Models - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    print("📚 This demo will:")
    print("1. Create a synthetic dataset")
    print("2. Train a simple BERT model")
    print("3. Generate explanations using SHAP, LIME, and attention")
    print("4. Create visualizations")
    print("5. Launch the interactive demo")
    
    print("\n⚠️  DISCLAIMER: This is for research/education only!")
    print("   Do not use for regulated decisions without human review.")
    
    response = input("\n🤔 Continue? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("👋 Goodbye!")
        sys.exit(0)
    
    # Run the simple example
    print("\n🔄 Running simple example...")
    try:
        result = subprocess.run([sys.executable, "examples/simple_example.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ Simple example completed successfully!")
        print("\n" + "="*50)
        print("📊 RESULTS:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Simple example failed: {e}")
        print(f"STDERR: {e.stderr}")
        return
    
    # Ask about launching demo
    print("\n🎯 Would you like to launch the interactive Streamlit demo?")
    response = input("Launch demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Launching Streamlit demo...")
        print("📱 The demo will open in your browser")
        print("🔄 Use Ctrl+C to stop the demo")
        
        try:
            subprocess.run(["streamlit", "run", "demo/app.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed to launch: {e}")
        except KeyboardInterrupt:
            print("\n👋 Demo stopped by user")
    
    print("\n✅ Quick start completed!")
    print("\n📚 What's next?")
    print("- Read README.md for detailed documentation")
    print("- Check DISCLAIMER.md for important limitations")
    print("- Run 'python scripts/train.py --help' for training options")
    print("- Explore the examples/ directory for more demos")


if __name__ == "__main__":
    main()
