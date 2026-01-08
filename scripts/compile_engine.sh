#!/bin/bash

# Template for compiling ConeNet-DLA on NVIDIA Jetson Orin AGX
# Based on RESEARCH.md Section 6.2

INPUT_ONNX="work_dirs/conenet_qat_sparse.onnx"
OUTPUT_ENGINE="work_dirs/conenet_dla.engine"

echo "Compiling ConeNet for DLA Core 0..."

trtexec --onnx=$INPUT_ONNX \
        --saveEngine=$OUTPUT_ENGINE \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --sparsity=enable \
        --allowGPUFallback=false \
        --inputIOFormats=int8:chw \
        --outputIOFormats=fp16:chw \
        --profilingVerbosity=detailed

echo "Compilation complete. Engine saved to $OUTPUT_ENGINE"
echo "Note: Ensure allowGPUFallback=false to guarantee zero-latency determinism."
