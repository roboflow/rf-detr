import sys
import re
from pathlib import Path

# Apache 2.0 RF-DETR header pattern
APACHE_PATTERN = re.compile(
    r"# RF-DETR\s*\n# Copyright \(c\) 2025 Roboflow\. All Rights Reserved\.\s*\n# Licensed under the Apache License, Version 2\.0"
)

# Platform Model License 1.0 pattern
PML_PATTERN = re.compile(
    r"# Platform Model License 1\.0 \(PML-1\.0\)\s*\n# Copyright \(c\) 2026 Roboflow, Inc\. All Rights Reserved\."
)

def check_file(path):
    try:
        content = Path(path).read_text()
        header = "\n".join(content.splitlines()[:15])
        
        if APACHE_PATTERN.search(header) or PML_PATTERN.search(header):
            return True
        return False
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return False

def main():
    files = sys.argv[1:]
    failed_files = []
    
    for file in files:
        if not check_file(file):
            failed_files.append(file)
            
    if failed_files:
        print("❌ The following files are missing valid license headers:")
        for file in failed_files:
            print(f"  - {file}")
        print("\nEach Python file must start with one of the following headers:")
        print("\n1. Apache 2.0 (RF-DETR):")
        print("   # ------------------------------------------------------------------------")
        print("   # RF-DETR")
        print("   # Copyright (c) 2025 Roboflow. All Rights Reserved.")
        print("   # Licensed under the Apache License, Version 2.0 [see LICENSE for details]")
        print("   # ------------------------------------------------------------------------")
        print("\n2. Platform Model License 1.0:")
        print("   # ------------------------------------------------------------------------")
        print("   # Platform Model License 1.0 (PML-1.0)")
        print("   # Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.")
        print("   # ...")
        print("   # ------------------------------------------------------------------------")
        sys.exit(1)
    
    print("✅ All Python files have valid license headers.")

if __name__ == "__main__":
    main()
