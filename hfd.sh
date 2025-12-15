#!/bin/bash
# HuggingFace Downloader Script
# Downloads files from HuggingFace Hub using huggingface_hub

set -e

REPO_ID="$1"
shift

if [ -z "$REPO_ID" ]; then
    echo "Usage: hfd.sh <repo_id> [--include <file1>] [--include <file2>] [--local-dir <dir>] [--tool <tool>]"
    exit 1
fi

# Parse arguments
INCLUDE_FILES=()
LOCAL_DIR=""
TOOL="hf_transfer"

while [[ $# -gt 0 ]]; do
    case $1 in
        --include)
            INCLUDE_FILES+=("$2")
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --tool)
            TOOL="$2"
            shift 2
            ;;
        -x|--max-connections)
            shift 2
            ;;
        -j|--jobs)
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default local directory if not provided
if [ -z "$LOCAL_DIR" ]; then
    LOCAL_DIR="/tmp/hfd_$(basename "$REPO_ID" | tr '/' '_')"
fi

# Create local directory
mkdir -p "$LOCAL_DIR"

# Export HF_HOME for caching
export HF_HOME="${HF_HOME:-/home/user/.cache/huggingface}"

# Build Python array string for include files
INCLUDE_JSON="[]"
if [ ${#INCLUDE_FILES[@]} -gt 0 ]; then
    INCLUDE_JSON=$(python3 -c "import json; import sys; files = sys.argv[1:]; print(json.dumps(files))" "${INCLUDE_FILES[@]}")
fi

# Download files using huggingface_hub Python library
python3 << EOF
import os
import sys
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

repo_id = "$REPO_ID"
local_dir = "$LOCAL_DIR"
include_files = json.loads('$INCLUDE_JSON')

try:
    if include_files:
        # Download specific files
        for file_pattern in include_files:
            # Handle subfolder paths (e.g., "pose/dw-ll_ucoco_384.onnx")
            if "/" in file_pattern:
                subfolder = os.path.dirname(file_pattern)
                filename = os.path.basename(file_pattern)
                target_path = os.path.join(local_dir, subfolder)
                os.makedirs(target_path, exist_ok=True)
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=target_path,
                    subfolder=subfolder if subfolder else None
                )
                print(f"Downloaded: {downloaded_path}")
            else:
                # Simple filename
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_pattern,
                    local_dir=local_dir
                )
                print(f"Downloaded: {downloaded_path}")
    else:
        # Download entire repository
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded entire repository to: {local_dir}")
    
    print("✓ Download completed successfully")
except Exception as e:
    print(f"✗ Download failed: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

