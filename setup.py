"""
Setup and installation script for the E-Commerce Recommendation System.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages from requirements.txt"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if requirements_file.exists():
        print("Installing requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            return False
    else:
        print("❌ requirements.txt not found!")
        return False
    
    return True


def create_output_directory():
    """Create output directory for results"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"✅ Output directory created: {output_dir}")


def setup_environment():
    """Setup the environment for the recommendation system"""
    print("🚀 Setting up E-Commerce Recommendation System...")
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create output directory
    create_output_directory()
    
    print("✅ Setup completed successfully!")
    print("\nTo run the system:")
    print("python main.py")
    
    return True


if __name__ == "__main__":
    setup_environment()
