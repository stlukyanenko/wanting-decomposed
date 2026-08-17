"""Modal app: run battery / extraction on a cloud GPU, results returned to src/runs/.

Model weights cache on the persistent dm-hf-cache volume (download once). Code is the
same src/ modules that ran locally — nothing is debugged remotely (D26).

Usage:
  modal run src/modal_app.py::battery -- --model Qwen/Qwen3.5-9B --out gate9b_a0.json
  modal run src/modal_app.py::exec_cmd -- --cmd "python /root/code/extraction.py stories --model Qwen/Qwen3.5-9B"
Outputs land in src/runs/ locally (returned as bytes) AND persist on the runs volume.
"""

import json
from pathlib import Path

import modal

app = modal.App("dm-affect")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers==5.15.0", "accelerate", "numpy", "scipy")
    .env({"HF_HOME": "/hf"})
    # ignore volatile outputs: locally-landing battery JSONs and judge files change
    # mid-hash when jobs run in parallel and abort concurrent launches ("modified
    # during build"). The container only needs code + runs/extraction (dials).
    .add_local_dir(str(Path(__file__).parent), "/root/code",
                   ignore=["runs/*.json", "runs/probes/**", "runs/preambles/**",
                           "runs/*.jsonl", "__pycache__/**", ".venv/**"])
)
hf_cache = modal.Volume.from_name("dm-hf-cache", create_if_missing=True)

# Optional HF token (gated models like Gemma). Create with:
#   modal secret create hf-token HF_TOKEN=hf_...
try:
    _secrets = [modal.Secret.from_name("hf-token")]
except Exception:
    _secrets = []
runs_vol = modal.Volume.from_name("dm-runs", create_if_missing=True)

import os

GPU = os.environ.get("DM_GPU", "A10G")
# A10G (24GB) fits Qwen3.5-9B bf16. For Gemma-4-12B (~24GB weights alone) run the
# client with DM_GPU=L40S (48GB; requires a payment method on the Modal account).


@app.function(gpu=GPU, image=image, timeout=7200, secrets=_secrets,
              volumes={"/hf": hf_cache, "/runs": runs_vol})
def run_battery(argv: list[str]) -> bytes:
    import subprocess
    import sys

    out_name = argv[argv.index("--out") + 1]
    remote_out = f"/runs/{out_name}"
    argv = list(argv)
    argv[argv.index("--out") + 1] = remote_out
    cmd = [sys.executable, "/root/code/battery.py"] + argv
    print("exec:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd="/root/code")
    runs_vol.commit()
    return Path(remote_out).read_bytes()


@app.function(gpu=GPU, image=image, timeout=7200, secrets=_secrets,
              volumes={"/hf": hf_cache, "/runs": runs_vol})
def run_cmd(cmd: str) -> str:
    """Escape hatch for extraction phases; artifacts persist on the runs volume."""
    import subprocess

    # Stream output live (don't capture) so local logs show progress in real time.
    r = subprocess.run(cmd, shell=True, cwd="/root/code")
    runs_vol.commit()
    if r.returncode != 0:
        raise RuntimeError(f"remote command failed with code {r.returncode}")
    return "ok"


@app.local_entrypoint()
def battery(model: str, out: str, alpha: float = 0.0, template: int = 1,
            max_items: int = 0, batch_size: int = 64, vector: str = "",
            layer: int = -1, system: str = "", preamble_file: str = ""):
    argv = ["--model", model, "--out", out, "--alpha", str(alpha),
            "--template", str(template), "--batch-size", str(batch_size),
            "--pool", "/root/code/data/pool.json"]
    if max_items:
        argv += ["--max-items", str(max_items)]
    if vector:
        argv += ["--vector", vector]
    if layer >= 0:
        argv += ["--layer", str(layer)]
    if system:
        argv += ["--system", system]
    if preamble_file:
        argv += ["--preamble-file", preamble_file]
    data = run_battery.remote(argv)
    local = Path(__file__).parent / "runs" / out
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    meta = json.loads(data)
    print(f"saved {local} | elicit {meta['clock']['elicit_s']}s | "
          f"low-mass {meta['low_mass_prompts']}")


@app.local_entrypoint()
def exec_cmd(cmd: str):
    print(run_cmd.remote(cmd))
