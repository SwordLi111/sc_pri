"""Stage 1: Download OI-V7 subset for face + plate (bbox only)."""
import os

import fiftyone as fo
import fiftyone.zoo as foz

from sc_pri.utils import load_config


def download_split(split, max_samples, dataset_name, classes, seed):
    if fo.dataset_exists(dataset_name):
        print(f"[{split}] existing dataset '{dataset_name}' found, deleting...")
        fo.delete_dataset(dataset_name)

    ds = foz.load_zoo_dataset(
        "open-images-v7",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=max_samples,
        seed=seed,
        shuffle=True,
        dataset_name=dataset_name,
        only_matching=True,
    )
    ds.persistent = True
    return ds


def report(ds, name):
    print(f"\n=== {name} ({len(ds)} samples) ===")
    print("Fields:", list(ds.get_field_schema().keys()))

    try:
        counts = ds.count_values("detections.detections.label")
        print("Detection counts:")
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {n}")
    except Exception as e:
        print(f"(detection counts error: {e})")


def main():
    cfg = load_config("configs/data.yaml")

    zoo_dir = os.path.expanduser(cfg["download"]["fiftyone_zoo_dir"])
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = zoo_dir
    os.makedirs(zoo_dir, exist_ok=True)
    print(f"fiftyone zoo dir: {zoo_dir}")

    classes = [c["name"] for c in cfg["classes"]]
    seed = cfg["download"]["seed"]

    train = download_split(
        "train", cfg["download"]["n_train"],
        "oi_face_plate_train", classes, seed,
    )
    val = download_split(
        "validation", cfg["download"]["n_val"],
        "oi_face_plate_val", classes, seed,
    )

    report(train, "TRAIN")
    report(val, "VAL")

    print("\nDone. Reload later via:")
    print('  fo.load_dataset("oi_face_plate_train")')


if __name__ == "__main__":
    main()