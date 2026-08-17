"""Minimal Modal sanity check: can we get a GPU, see CUDA, and touch a persistent volume?

Run:  modal run src/modal_check.py
Cost: ~1 GPU-minute on A10G (cents).
"""

import modal

app = modal.App("dm-affect-check")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")
hf_cache = modal.Volume.from_name("dm-hf-cache", create_if_missing=True)


@app.function(gpu="A10G", image=image, timeout=300, volumes={"/hf": hf_cache})
def gpu_check() -> dict:
    import torch

    x = torch.randn(2048, 2048, device="cuda")
    y = (x @ x).sum().item()
    with open("/hf/.probe", "w") as f:
        f.write("volume writable")
    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        "matmul_ok": bool(y == y),  # not-NaN
        "volume": open("/hf/.probe").read(),
    }


@app.local_entrypoint()
def main():
    print(gpu_check.remote())
