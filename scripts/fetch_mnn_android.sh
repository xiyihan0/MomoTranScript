#!/usr/bin/env bash
set -euo pipefail

# Download official Alibaba MNN 3.6.2 Android prebuilt libraries (CPU + OpenCL + Vulkan)
# and stage arm64-v8a .so files into app/src/main/jniLibs/arm64-v8a/.

MNN_VER="3.6.2"
ARCHIVE="mnn_${MNN_VER}_android_armv7_armv8_cpu_opencl_vulkan.zip"
URL="https://github.com/alibaba/MNN/releases/download/${MNN_VER}/${ARCHIVE}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JNI_DIR="${ROOT_DIR}/app/src/main/jniLibs/arm64-v8a"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

echo "Downloading official MNN ${MNN_VER} Android prebuilt..."
curl -fL --retry 3 -o "${TMP_DIR}/${ARCHIVE}" "${URL}"
unzip -q "${TMP_DIR}/${ARCHIVE}" -d "${TMP_DIR}/mnn"

mkdir -p "${JNI_DIR}"
SRC_ARM64="$(find "${TMP_DIR}/mnn" -type d -name "arm64-v8a" | head -n 1)"
if [[ -z "${SRC_ARM64}" ]]; then
  echo "ERROR: arm64-v8a directory not found inside ${ARCHIVE}" >&2
  exit 1
fi

for lib in libMNN.so libMNN_Express.so libMNN_CL.so libMNN_Vulkan.so; do
  cp -f "${SRC_ARM64}/${lib}" "${JNI_DIR}/${lib}"
done

ls -lh "${JNI_DIR}"/libMNN*.so
echo "MNN ${MNN_VER} arm64-v8a libraries staged successfully."
