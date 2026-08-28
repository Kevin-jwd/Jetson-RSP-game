"""Merge YOLO detection datasets into one, remapping class ids by name.

Public rock-paper-scissors datasets order their classes alphabetically
(``Paper, Rock, Scissors``) while this game uses ``scissors, rock, paper``. Ids
are remapped by class *name*, so mixing sources cannot silently swap rock for
paper — the failure that looks like a badly trained model rather than a bug.

Layout is detected per source: ``images/train`` + ``labels/train`` (the class
dataset) or ``train/images`` + ``train/labels`` (Roboflow). ``valid`` is renamed
to ``val``; every other split keeps its name.

    python tools/merge_dataset.py --out RPS_Merged \\
        --src RPS_Dataset_YOLO --src rock-paper-scissors-sxsw-14
"""

import argparse
import shutil
from pathlib import Path

# The order the game and the engine expect; see CLASS_NAMES in rps/detector.py.
TARGET = ["scissors", "rock", "paper"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"valid": "val", "validation": "val"}


def source_names(root: Path) -> list[str]:
    """Class order from the source's data.yaml, defaulting to TARGET."""
    for yaml_path in (root / "data.yaml", root / "dataset.yaml"):
        if not yaml_path.exists():
            continue
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("names:") and "[" in line:
                raw = line.split("[", 1)[1].rsplit("]", 1)[0]
                return [n.strip().strip("'\"") for n in raw.split(",") if n.strip()]
    print(f"  no names in data.yaml, assuming {TARGET}")
    return list(TARGET)


def split_dirs(root: Path):
    """Yield (split, images_dir, labels_dir) for whichever layout this source uses."""
    for images in sorted(root.glob("images/*")):
        labels = root / "labels" / images.name
        if images.is_dir() and labels.is_dir():
            yield SPLIT_ALIASES.get(images.name, images.name), images, labels
    for images in sorted(root.glob("*/images")):
        labels = images.parent / "labels"
        if images.is_dir() and labels.is_dir():
            yield SPLIT_ALIASES.get(images.parent.name, images.parent.name), images, labels


def remap_label(text: str, mapping: dict[int, int]) -> str:
    out = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        old = int(float(parts[0]))
        if old not in mapping:  # a class this game does not play
            continue
        out.append(" ".join([str(mapping[old])] + parts[1:]))
    return "\n".join(out) + ("\n" if out else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", action="append", required=True, help="dataset root (repeatable)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    counts: dict[str, int] = {}

    for src in args.src:
        root = Path(src)
        print(f"\n{root}")
        names = source_names(root)
        print(f"  names: {names}")

        # Match on lowercase names so 'Paper' and 'paper' are the same class.
        lowered = [n.lower() for n in names]
        mapping = {i: TARGET.index(n) for i, n in enumerate(lowered) if n in TARGET}
        dropped = [n for n in lowered if n not in TARGET]
        if dropped:
            print(f"  dropping classes not in {TARGET}: {dropped}")
        print(f"  id map: {mapping}")

        for split, images, labels in split_dirs(root):
            dst_img = out / "images" / split
            dst_lbl = out / "labels" / split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            n = 0
            for image in sorted(images.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                # Prefix with the source name: the same file name in two datasets
                # would otherwise silently overwrite.
                stem = f"{root.name}_{image.stem}"
                shutil.copy2(image, dst_img / f"{stem}{image.suffix}")

                label = labels / f"{image.stem}.txt"
                text = label.read_text(encoding="utf-8") if label.exists() else ""
                (dst_lbl / f"{stem}.txt").write_text(remap_label(text, mapping), encoding="utf-8")
                n += 1

            counts[split] = counts.get(split, 0) + n
            print(f"  {split}: {n} images")

    yaml = out / "data.yaml"
    val = "val" if (out / "images/val").exists() else "test"
    yaml.write_text(
        f"train: images/train\nval: images/{val}\n"
        + (f"test: images/test\n" if (out / "images/test").exists() else "")
        + f"\nnc: {len(TARGET)}\nnames: {TARGET}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {yaml}")
    for split, n in sorted(counts.items()):
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
