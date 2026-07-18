"""Stage 2: Cache latents + multi-class masks."""
import os
from pathlib import Path

from sc_pri.utils import load_config
from sc_pri.vae import load_vae
from sc_pri.data.oi_cache import process_split


def main():
    cfg = load_config("configs/data.yaml")
    
    # Set fiftyone zoo dir (must match download step)
    zoo_dir = os.path.expanduser(cfg["download"]["fiftyone_zoo_dir"])
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = zoo_dir
    
    class_to_idx = {c["name"]: c["idx"] for c in cfg["classes"]}
    print(f"class_to_idx = {class_to_idx}")
    
    device = cfg["vae"]["device"]
    vae = load_vae(cfg["vae"]["model_id"], device=device)
    print(f"VAE loaded on {device}")
    
    cache_root = cfg["cache"]["root"]
    
    process_split(
        "oi_face_plate_train",
        os.path.join(cache_root, cfg["cache"]["train_subdir"]),
        vae, cfg, class_to_idx,
    )
    process_split(
        "oi_face_plate_val",
        os.path.join(cache_root, cfg["cache"]["val_subdir"]),
        vae, cfg, class_to_idx,
    )
    
    print("\nCache complete.")


if __name__ == "__main__":
    main()