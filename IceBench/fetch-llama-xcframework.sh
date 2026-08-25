#!/bin/bash
# Fetches the prebuilt llama.cpp xcframework used by IceBench's llama.cpp
# comparison arm. Run once before opening the project.
set -euo pipefail
cd "$(dirname "$0")"
BUILD=b10612
if [ -d llama.xcframework ]; then echo "llama.xcframework already present"; exit 0; fi
echo "Downloading llama.cpp $BUILD xcframework (~80 MB)…"
curl -sL -o /tmp/llama-xcframework.zip "https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-xcframework.zip"
unzip -q /tmp/llama-xcframework.zip -d /tmp/llama-xcframework-extract
mv /tmp/llama-xcframework-extract/build-apple/llama.xcframework .
rm -rf /tmp/llama-xcframework.zip /tmp/llama-xcframework-extract
echo "Done: $(pwd)/llama.xcframework"
