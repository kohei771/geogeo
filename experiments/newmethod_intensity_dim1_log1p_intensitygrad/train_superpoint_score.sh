#!/bin/bash

# SuperPoint Score Training Script
# This script trains the superpoint score module separately from the main model

echo "Starting SuperPoint Score Training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:../../"

# Run the training
python superpoint_score_separate_train.py

echo "SuperPoint Score Training completed!"
echo "Weights saved to: score_weights.pth"
