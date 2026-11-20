#!/bin/bash
# Recompile the per-platform locks from desktop/requirements.in (needs uv; any OS).
set -euo pipefail
cd "$(dirname "$0")"
common=(--python-version 3.12 --only-binary :all: --no-binary sheetsage2-refined --no-binary muscriptor
        --no-binary chord-romanizer --no-binary dlchordx -q)
uv pip compile ../requirements.in "${common[@]}" --python-platform x86_64-pc-windows-msvc \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match -o windows.txt
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile ../requirements.in "${common[@]}" \
  --python-platform aarch64-apple-darwin -o macos.txt
echo "windows: $(grep -c '==' windows.txt) pins, $(grep -o '^torch==[^ ]*' windows.txt)"
echo "macos:   $(grep -c '==' macos.txt) pins, $(grep -o '^torch==[^ ]*' macos.txt)"
