"""Create a truncated (n-layer) Qwen2 model dir from a cached full model.

Keeps: embeddings, lm_head, final norm, and layers 0..nlayer-1.
Rewrites config.json with num_hidden_layers=nlayer. Uses memory-mapped
slices so the full 3.5GB checkpoint never needs to fit in RAM.

Usage:
    source venv/bin/activate
    python scripts/make_truncated_model.py \
        --src /home/ubuntu/.cache/huggingface/hub/models--.../snapshots/<hash> \
        --dst /home/ubuntu/code/models_truncated/qwen2-1.5b-2l \
        --nlayer 2
Run test_infer on project root directory:
    PYTHONPATH=test ./venv/bin/python test/test_infer.py --test --model .cache/models/qwen2-1.5b-2l --max_steps 32
"""
import argparse
import json
import shutil
from pathlib import Path

import safetensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--dst", type=str, required=True)
    ap.add_argument("--nlayer", type=int, default=2)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # copy config + tokenizer files, rewrite layer count
    for name in ["tokenizer_config.json", "tokenizer.json", "generation_config.json"]:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    config = json.loads((src / "config.json").read_text())
    config["num_hidden_layers"] = args.nlayer
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    src_files = sorted(src.glob("*.safetensors"))
    print(f"src files: {[f.name for f in src_files]}")

    keep_prefix = set()
    for i in range(args.nlayer):
        keep_prefix.add(f"model.layers.{i}.")
    keep_exact = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}

    n_kept = 0
    with safetensors.safe_open(src_files[0], framework="pt", device="cpu") as f:
        tensors = {}
        for key in f.keys():
            if key in keep_exact or any(key.startswith(p) for p in keep_prefix):
                tensors[key] = f.get_slice(key)  # lazy mmap slice
                n_kept += 1
        # materialize into dict of arrays (only kept tensors, ~small)
        loaded = {k: v[:] for k, v in tensors.items()}
        dtype_map = {"BFLOAT16": "bfloat16", "FLOAT16": "float16", "FLOAT32": "float32"}
        specs = {}
        for key, t in loaded.items():
            s = str(t.dtype).split(".")[-1].upper()
            dtype = dtype_map.get(s, s)
            specs[key] = safetensors.TensorSpec(
                dtype=dtype,
                shape=tuple(t.shape),
                data_ptr=t.data_ptr(),
                data_len=t.numel() * t.element_size(),
            )
        out_path = dst / "model.safetensors"
        safetensors.serialize_file(specs, str(out_path))
        print(f"kept {n_kept} tensors -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
