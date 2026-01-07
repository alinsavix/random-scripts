"""
Debug script to help identify Backblaze data files and their locations
"""

import os
import glob

def find_backblaze_files():
    """Search for Backblaze installation and data files"""
    print("=" * 80)
    print("BACKBLAZE DEBUG INFORMATION")
    print("=" * 80)

    # Check for Backblaze installation
    print("\n1. Checking Backblaze Installation Paths:")
    install_paths = [
        r"C:\Program Files (x86)\Backblaze",
        r"C:\Program Files\Backblaze",
    ]

    for path in install_paths:
        if os.path.exists(path):
            print(f"   ✓ Found: {path}")
            # List executables
            exes = glob.glob(os.path.join(path, "*.exe"))
            for exe in exes:
                print(f"     - {os.path.basename(exe)}")
        else:
            print(f"   ✗ Not found: {path}")

    # Check for Backblaze data directories
    print("\n2. Checking Backblaze Data Directories:")
    data_paths = [
        os.path.join(os.environ.get('PROGRAMDATA', 'C:\\ProgramData'), 'Backblaze'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Backblaze'),
        os.path.join(os.environ.get('APPDATA', ''), 'Backblaze'),
    ]

    for path in data_paths:
        if os.path.exists(path):
            print(f"   ✓ Found: {path}")
            # List subdirectories
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        print(f"     - [DIR] {item}")
                    else:
                        print(f"     - [FILE] {item}")
            except Exception as e:
                print(f"     Error listing contents: {e}")
        else:
            print(f"   ✗ Not found: {path}")

    # Check for specific data files
    print("\n3. Looking for Backblaze Data Files:")
    bz_data = os.path.join(os.environ.get('PROGRAMDATA', 'C:\\ProgramData'), 'Backblaze')

    if os.path.exists(bz_data):
        search_patterns = [
            'bzdata\\bzbackup\\*.txt',
            'bzdata\\bzbackup\\*.xml',
            'bzdata\\bzbackup\\*.dat',
            'bzdata\\bztransmit\\*.log',
            'bzlogs\\*.log',
        ]

        for pattern in search_patterns:
            full_pattern = os.path.join(bz_data, pattern)
            files = glob.glob(full_pattern)
            if files:
                print(f"\n   Pattern: {pattern}")
                for f in files[:10]:  # Limit to first 10
                    rel_path = os.path.relpath(f, bz_data)
                    size = os.path.getsize(f)
                    print(f"     - {rel_path} ({size:,} bytes)")
                if len(files) > 10:
                    print(f"     ... and {len(files) - 10} more files")

    # Check service status
    print("\n4. Checking Backblaze Service Status:")
    import subprocess
    try:
        result = subprocess.run(
            ['sc', 'query', 'Backblaze Backup Service'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✓ Backblaze Backup Service found")
            if 'RUNNING' in result.stdout:
                print("   ✓ Service is RUNNING")
            elif 'STOPPED' in result.stdout:
                print("   ✗ Service is STOPPED")
            else:
                print(f"   ? Service status: {result.stdout.split('STATE')[1].split()[0] if 'STATE' in result.stdout else 'Unknown'}")
        else:
            print("   ✗ Backblaze Backup Service not found")
    except Exception as e:
        print(f"   ✗ Error checking service: {e}")

    print("\n" + "=" * 80)
    print("Debug information complete!")
    print("=" * 80)

if __name__ == "__main__":
    find_backblaze_files()
    input("\nPress Enter to exit...")
