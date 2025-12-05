# Remote FLUX.2 Text Encoder (HuggingFace) – ComfyUI Custom Node

A lightweight ComfyUI node that sends your prompts to the HuggingFace FLUX.2 Remote Text Encoder and returns FLUX.2-compatible conditioning tensors.
No local text encoder, no heavy model downloads — remote, fast, and ready to plug into any FLUX.2 workflow.

## 🔹 Purpose

This node replaces the local FLUX.2 text encoder with a remote HuggingFace-hosted encoder, so you can:

Generate embeddings without loading large models

Reduce VRAM usage

Improve startup speed

Use the official remote encoder format fully compatible with ComfyUI + FLUX.2

## 🔹 Installation
Option 1 — Clone directly into ComfyUI
## cd path/to/ComfyUI/custom_nodes
```bash
git clone https://github.com/vimal-v-2006/ComfyUI-Remote-FLUX2-Text-Encoder-HuggingFace.git


Restart ComfyUI.
The node auto-installs required packages (requests, huggingface_hub) on first load.

## Option 2 — Manual install

Download remote_flux2_text_encoder_hf.py

Place it into:

ComfyUI/custom_nodes/


Restart ComfyUI
The node auto-installs dependencies automatically.

## 🔹 Create a HuggingFace Token (step-by-step)

Log in to: https://huggingface.co

Go to Profile → Settings → Access Tokens

Click New Token

Choose Read permission

Copy the token (hf_...)

Keep it private. Do not upload it to GitHub.

## 🔹 How to Use (simple steps)

Open ComfyUI

Add the node:
Remote FLUX.2 Text Encoder (HF)
(found under: conditioning/flux2_remote)

Fill fields:

Positive Prompt → your main prompt

Negative Prompt (optional)

## **HF API Token → paste your hf_... token**

Leave Endpoint URL and Device as default

Connect outputs:

positive → FLUX.2 conditioning input

negative → negative conditioning input (optional)

Generate your image
The node sends your text to HuggingFace and returns correct FLUX.2 embeddings.
