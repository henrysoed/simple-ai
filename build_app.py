"""
Build script for preparing the app for deployment.
This script creates a clean deployment folder with only necessary files.
"""
import os
import shutil
import sys

# Configuration
DEPLOY_DIR = "deploy"
SOURCE_DIR = "."
NECESSARY_FILES = [
    "app.py",
    "requirements.txt",
    "vercel.json",
    ".vercelignore",
    "templates"
]

# Optional files that might be needed for reference but not critical
OPTIONAL_FILES = [
    "README.md"
]

def clean_deploy_dir():
    """Create a clean deployment directory."""
    # Remove old deploy dir if exists
    if os.path.exists(DEPLOY_DIR):
        print(f"Removing existing {DEPLOY_DIR} directory...")
        shutil.rmtree(DEPLOY_DIR)

    # Create new deploy dir
    print(f"Creating new {DEPLOY_DIR} directory...")
    os.makedirs(DEPLOY_DIR)
    os.makedirs(os.path.join(DEPLOY_DIR, "templates"), exist_ok=True)

def copy_necessary_files():
    """Copy only necessary files to the deployment directory."""
    for file in NECESSARY_FILES:
        source_path = os.path.join(SOURCE_DIR, file)
        dest_path = os.path.join(DEPLOY_DIR, file)
        
        if os.path.isfile(source_path):
            print(f"Copying {file}...")
            shutil.copy2(source_path, dest_path)
        elif os.path.isdir(source_path):
            print(f"Copying directory {file}...")
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
        else:
            print(f"Warning: {file} not found!")

    # Copy optional files
    for file in OPTIONAL_FILES:
        source_path = os.path.join(SOURCE_DIR, file)
        if os.path.exists(source_path):
            dest_path = os.path.join(DEPLOY_DIR, file)
            print(f"Copying optional file {file}...")
            shutil.copy2(source_path, dest_path)

def main():
    """Main function to prepare the deployment."""
    try:
        clean_deploy_dir()
        copy_necessary_files()
        print("\nDeployment preparation complete!")
        print(f"Your cleaned application is ready in the '{DEPLOY_DIR}' folder.")
        print("You can now deploy this folder to Vercel.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
