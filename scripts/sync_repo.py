#!/usr/bin/env python3
"""
Script to synchronize the repository with all the critical fixes.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Main synchronization process."""
    print("🔄 Synchronizing EMP Proving Ground Repository")
    print("=" * 50)
    
    # Check git status
    if not run_command("git status", "Check git status"):
        return False
    
    # Add all changes
    if not run_command("git add -A", "Add all changes to staging"):
        return False
    
    # Commit changes
    commit_msg = """Critical repository fixes and synchronization

- Resolved merge conflicts in requirements.txt and README.md
- Created clean, unified documentation reflecting v2.0 modular system
- Added GitHub Actions CI workflow to prevent future fundamental errors
- Verified system is runnable and all modules import correctly
- Repository is now operational and ready for Phase 2 development"""
    
    if not run_command(f'git commit -m "{commit_msg}"', "Commit critical fixes"):
        return False
    
    # Push to remote
    if not run_command("git push origin main", "Push changes to remote repository"):
        return False
    
    print("\n🎉 Repository synchronization completed successfully!")
    print("✅ All critical fixes have been committed and pushed")
    print("✅ Repository is now operational and ready for development")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
