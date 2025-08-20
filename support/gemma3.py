#!/usr/bin/env python3
"""
Convert Gemma 3 model from HuggingFace (SafeTensors) to ONNX format
"""
import os
import sys
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_gemma_to_onnx(model_id, output_dir, token=None):
    """
    Convert a Gemma model from HuggingFace to ONNX format

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-3-270m-it")
        output_dir: Directory to save the ONNX model
        token: HuggingFace auth token for gated models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Converting {model_id} to ONNX format...")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    # Convert to ONNX using Optimum
    logger.info("Converting to ONNX...")
    onnx_model = ORTModelForCausalLM.from_pretrained(
        # model,
        model_id,
        from_transformers=True,
        export=True,
        provider="CPUExecutionProvider",
        torch_dtype=torch.float16,
        device_map="auto",
        # output=str(output_dir)
    )
    logger.info("Loaded model")
    onnx_model.save_pretrained(output_dir)
    logger.info("Saved model")

    # Save tokenizer alongside the model
    tokenizer.save_pretrained(output_dir)

    logger.info(f"✅ Model successfully exported to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gemma_to_onnx.py <model_id> <output_dir> [auth_token]")
        print("Example: python gemma_to_onnx.py google/gemma-3-270m-it ../models/gemma-onnx YOUR_HF_TOKEN_HERE")
        sys.exit(1)

    model_id = sys.argv[1]
    output_dir = sys.argv[2]
    auth_token = sys.argv[3] if len(sys.argv) > 3 else None

    convert_gemma_to_onnx(model_id, output_dir, auth_token)
