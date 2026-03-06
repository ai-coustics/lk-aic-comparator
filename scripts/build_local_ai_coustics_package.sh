#!/usr/bin/env bash
set -euo pipefail

PLUGINS_REPO="${1:-$HOME/dev/projects/plugins-ai-coustics-internal}"
UNIFFI_DIR="$PLUGINS_REPO/crates/plugins-ai-coustics-uniffi"
TARGET_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
PACKAGE_DIR="$PLUGINS_REPO/target/packages/python"
LIB_PATH="$PLUGINS_REPO/target/$TARGET_TRIPLE/release/libplugins_ai_coustics_uniffi.so"

cd "$UNIFFI_DIR"

rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

export CARGO_PROFILE_RELEASE_STRIP=false
cargo build --release --features skip-auth --target "$TARGET_TRIPLE"

cp support/python/.gitignore "$PACKAGE_DIR/"
cp support/python/.python-version "$PACKAGE_DIR/"
cp support/python/README.md "$PACKAGE_DIR/"
cp support/python/setup.py "$PACKAGE_DIR/"
cp support/python/uv.lock "$PACKAGE_DIR/"
cp support/python/MANIFEST.in "$PACKAGE_DIR/"
cp -r support/python/src "$PACKAGE_DIR/"

cat > "$PACKAGE_DIR/pyproject.toml" << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "livekit-plugins-ai-coustics"
version = "0.2.2"
description = "LiveKit AI Coustics noise filtering plugin"
readme = "README.md"
requires-python = ">=3.9"
dependencies = ["livekit>=1.0.25", "livekit-agents>=1.4.2"]
keywords = ["webrtc", "realtime", "audio", "livekit", "ai-coustics"]
license = { text = "SEE LICENSE IN https://livekit.io/legal/terms-of-service" }

[tool.setuptools]
package-dir = { "" = "src" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"livekit.plugins.ai_coustics" = ["py.typed", "*.so", "*.dylib", "*.dll"]
EOF

cargo run --bin uniffi-bindgen -- generate \
  --language python \
  --out-dir "$PACKAGE_DIR" \
  --library "$LIB_PATH"

mv "$PACKAGE_DIR/plugins_ai_coustics_uniffi.py" \
  "$PACKAGE_DIR/src/livekit/plugins/ai_coustics/_ffi.py"
cp "$LIB_PATH" "$PACKAGE_DIR/src/livekit/plugins/ai_coustics/"

echo "Built local package: $PACKAGE_DIR"
