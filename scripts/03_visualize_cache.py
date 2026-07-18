"""Stage 3: Visualize cached latents + masks. CRITICAL sanity check."""
import os

from sc_pri.utils import load_config
from sc_pri.vae import load_vae
from sc_pri.viz.sanity import visualize_cached_samples


def main():
    cfg = load_config("configs/data.yaml")
    
    vae = load_vae(cfg["vae"]["model_id"], device=cfg["vae"]["device"])
    
    cache_train = os.path.join(cfg["cache"]["root"], cfg["cache"]["train_subdir"])
    out_dir = "debug/vis_train"
    
    visualize_cached_samples(
        cache_train, out_dir, vae,
        n_samples=20,
        scale_factor=cfg["latent"]["scale_factor"],
    )


if __name__ == "__main__":
    main()