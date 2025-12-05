# --- Auto-install light dependencies if missing (requests, huggingface_hub) ---
def _ensure_deps():
    import importlib
    missing = []
    for pkg in ("requests", "huggingface_hub"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        import subprocess, sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception as e:
            print(f"[RemoteFlux2TextEncoderHF] Failed to auto-install {missing}: {e}")

_ensure_deps()
# --- End auto-install block ---

# remote_flux2_text_encoder_hf.py
import io
import json
import os

import requests
import torch
from huggingface_hub import get_token


HF_DEFAULT_URL = os.environ.get(
    "HF_FLUX2_REMOTE_ENCODER_URL",
    "https://remote-text-encoder-flux-2.huggingface.co/predict",
)


def call_flux2_remote_encoder(prompts, endpoint_url=None, api_token=None, device="cpu", timeout=60):
    """
    prompts: list[str] or str
    returns: torch.Tensor with same format as official remote_text_encoder()
    """

    if isinstance(prompts, str):
        payload = {"prompt": prompts}
    else:
        payload = {"prompt": prompts}

    url = endpoint_url or HF_DEFAULT_URL

    # get token: priority order -> explicit -> env -> hf token helper
    token = (
        api_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or get_token()
    )

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
    )
    resp.raise_for_status()

    # Response is a torch.save'd tensor
    prompt_embeds = torch.load(io.BytesIO(resp.content), map_location="cpu")

    # We keep it on CPU – DiSTorch / Comfy will move to GPU where needed
    # But we allow explicit device move if you really want:
    if device != "cpu":
        prompt_embeds = prompt_embeds.to(device)

    return prompt_embeds


def _make_cond_from_embedding(e: torch.Tensor | None, like: torch.Tensor | None = None):
    """
    Wrap HF embeddings into Comfy/Flux CONDITIONING format:
        [[tensor[B, T, D], {"pooled_output": None, "attention_mask": [B, T]}]]
    """
    if e is None:
        if like is not None:
            # match shape of the other embedding (e.g. negative uses positive as template)
            z = torch.zeros_like(like)
        else:
            # fully empty: tiny dummy tensor (same behaviour as your original code,
            # but now with proper dict so encode_adm doesn't crash)
            z = torch.zeros((1, 1, 1), dtype=torch.float32)
    else:
        z = e

    if z.ndim == 2:
        # [B, D] -> [B, 1, D]
        z = z.unsqueeze(1)

    B, T, D = z.shape

    # This is exactly the structure your working example had:
    # {'pooled_output': None, 'attention_mask': tensor([...])}
    attention_mask = torch.ones((B, T), dtype=torch.long, device=z.device)
    meta = {
        "pooled_output": None,          # matches the "thing it wanted" dump you posted
        "attention_mask": attention_mask,
    }

    return [[z, meta]]


class RemoteFlux2TextEncoderHF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "hf_endpoint_url": ("STRING", {"default": HF_DEFAULT_URL}),
                "hf_api_token": ("STRING", {"default": ""}),
                "device": (["cpu", "cuda:0", "cuda:1"], {"default": "cpu"}),
            },
        }

    # two CONDITIONING outputs (you’ll usually use only "positive")
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning/flux2_remote"

    def encode(
        self,
        positive_prompt,
        negative_prompt="",
        hf_endpoint_url=None,
        hf_api_token="",
        device="cpu",
    ):

        # 1) Completely empty – return tiny dummy conds, but WITH pooled_output + mask
        if not positive_prompt.strip() and not negative_prompt.strip():
            z = torch.zeros((1, 1, 1), dtype=torch.float32)
            empty_cond = _make_cond_from_embedding(z)
            return (empty_cond, empty_cond)

        pos_emb = None
        neg_emb = None

        # 2) Encode positive
        if positive_prompt.strip():
            pos_emb = call_flux2_remote_encoder(
                positive_prompt,
                endpoint_url=hf_endpoint_url,
                api_token=hf_api_token or None,
                device=device,
            )

        # 3) Encode negative
        if negative_prompt.strip():
            neg_emb = call_flux2_remote_encoder(
                negative_prompt,
                endpoint_url=hf_endpoint_url,
                api_token=hf_api_token or None,
                device=device,
            )

        # 4) Normalize dims: [B, D] -> [B, 1, D] (final safety – _make_cond also checks)
        def _normalize(e):
            if e is None:
                return None
            if e.ndim == 2:
                e = e.unsqueeze(1)
            return e

        pos_emb = _normalize(pos_emb)
        neg_emb = _normalize(neg_emb)

        # 5) Wrap into CONDITIONING objects
        pos_cond = _make_cond_from_embedding(pos_emb)
        # For negative, if None, use zeros_like(pos_emb) so shapes match
        neg_cond = _make_cond_from_embedding(neg_emb, like=pos_emb)

        return (pos_cond, neg_cond)


NODE_CLASS_MAPPINGS = {
    "RemoteFlux2TextEncoderHF": RemoteFlux2TextEncoderHF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteFlux2TextEncoderHF": "Remote FLUX.2 Text Encoder (HF)",
}
