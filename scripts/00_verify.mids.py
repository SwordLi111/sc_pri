"""Stage 0: Verify class MIDs and whether they have segmentation masks."""
import pandas as pd

CLASS_DESC_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-class-descriptions-boxable.csv"
)
SEG_CLASSES_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-classes-segmentation.txt"
)


def main():
    class_df = pd.read_csv(CLASS_DESC_URL, header=None,
                            names=["MID", "DisplayName"])
    print(f"Total boxable classes: {len(class_df)}")
    
    targets = class_df[
        class_df["DisplayName"].str.contains("face|plate", 
                                             case=False, na=False)
    ]
    print("\n=== Face/plate related classes ===")
    print(targets.to_string(index=False))
    
    seg_classes = pd.read_csv(SEG_CLASSES_URL, header=None, names=["MID"])
    seg_mids = set(seg_classes["MID"].values)
    print(f"\nTotal segmentation classes: {len(seg_mids)}")
    
    print("\n=== Mask availability for face/plate ===")
    for _, row in targets.iterrows():
        has_mask = row["MID"] in seg_mids
        tag = "HAS MASK" if has_mask else "BBOX only"
        print(f"  {row['DisplayName']:35s} ({row['MID']:12s}): {tag}")


if __name__ == "__main__":
    main()