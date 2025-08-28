import argparse
import os
import shutil
import yaml
import torch

# Adjust import roots if needed
from nets.nn import yolo_v8_n, yolo_v8_s, yolo_v8_m, yolo_v8_l, yolo_v8_x
from fedn_util import extract_weights_from_model, load_weights_into_model  # sanity check

ARCHS = {
    "n": yolo_v8_n,
    "s": yolo_v8_s,
    "m": yolo_v8_m,
    "l": yolo_v8_l,
    "x": yolo_v8_x,
}

def load_params():
    with open(os.path.join("utils", "args.yaml"), "r") as f:
        return yaml.safe_load(f)

def build_model(arch_key: str, nc: int):
    if arch_key not in ARCHS:
        raise ValueError(f"Unknown arch '{arch_key}'. Choose from {list(ARCHS)}")
    return ARCHS[arch_key](num_classes=nc)

def maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    if not ckpt_path:
        return False
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    # Handle common formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and hasattr(state["model"], "state_dict"):
        state = state["model"].state_dict()

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[seed] loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[seed]  - missing keys: {len(missing)} (ok if archs differ)")
    if unexpected:
        print(f"[seed]  - unexpected keys: {len(unexpected)} (ok if archs differ)")
    return True

def main():
    ap = argparse.ArgumentParser(description="Create FEDn seed weights (.npz) for YOLO model")
    ap.add_argument("--arch", default="n", choices=list(ARCHS), help="Model size: n/s/m/l/x")
    ap.add_argument("--pretrained", default="", help="Optional .pt checkpoint to warm-start from")
    ap.add_argument("--out", required=True, help="Output seed path, e.g. seed/seed.npz")
    args = ap.parse_args()

    params = load_params()
    # Number of classes is derived from args.yaml -> 'names'
    if "names" not in params or not isinstance(params["names"], dict):
        raise RuntimeError("utils/args.yaml must define a 'names' dict (class id -> name).")
    nc = len(params["names"].values())
    print(f"[seed] classes (nc): {nc}")

    # Build model and (optionally) load checkpoint
    model = build_model(args.arch, nc)
    used_ckpt = maybe_load_checkpoint(model, args.pretrained) if args.pretrained else False

    # Export to FEDn weight file using your helper (returns a temp path)
    tmp_obj = extract_weights_from_model(model)
    print(f"[seed] extracted to temp: {tmp_obj}")

    # Ensure output dir exists
    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Handle both path-like and BytesIO returns
    if isinstance(tmp_obj, (str, os.PathLike)):
        shutil.copy2(tmp_obj, out_path)
    else:
        # Assume file-like (e.g., BytesIO)
        try:
            # If it's a BytesIO, getvalue() is best
            data = tmp_obj.getvalue()
        except AttributeError:
            # Generic file-like: read from current position
            data = tmp_obj.read()
        with open(out_path, "wb") as f:
            f.write(data)

    print(f"[seed] wrote seed -> {out_path}")

    # Optional quick self-check: load back the seed
    try:
        load_weights_into_model(out_path, model)
        print("[seed] sanity load OK ✔")
    except Exception as e:
        print(f"[seed] sanity load FAILED: {e}")

    print(f"[seed] done. arch={args.arch}, pretrained={'yes' if used_ckpt else 'no'}")
if __name__ == "__main__":
    main()
