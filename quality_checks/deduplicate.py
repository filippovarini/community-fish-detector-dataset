"""
Adapted from GerardCB's script:
https://github.com/GerardCB/community-fish-detector-dataset/blob/gc/dedup/datasets/deduplicate.py

Finds duplicate images across and within the processed datasets (data/final), before they are merged.
Analytical only: it counts and lists duplicates, and does not remove anything.

A duplicate is the same photo: an exact copy, a re-encoded or resized copy, a stretched copy (e.g. a 640x640
Roboflow export), a small crop, or a copy with a text overlay. Different frames of the same video are not
looked for. Each image gets an SSCD copy-detection descriptor (Pizzi et al. 2022,
https://github.com/facebookresearch/sscd-copy-detection, model sscd_disc_mixup, the image resized to 320x320
without keeping its aspect ratio). Pairs whose descriptors have a cosine similarity of at least --threshold are
duplicates, and files with the same SHA-1 are exact copies. Calibration on 7 processed datasets (54k images):
- the best match between datasets that cannot share photos (other places and years) never exceeded 0.40
- 600 synthetic copies matched their original at >= 0.9 in 61% (resized to half and re-encoded at quality 50),
  62% (cropped up to 10% per side), 71% (text overlay) and 100% (stretched to 640x640, flipped) of cases, and at
  >= 0.85 in 79%, 93%, 99% and 100%; a crop plus stretch plus a black text banner reached 0.9 in 1%, 0.7 in 88%
- neighbouring frames of a video (1 s apart) reach 0.75-0.9, and frames of a fixed camera often exceed 0.95;
  views of one fixed-camera scene on different days reached 0.87

For every image, three neighbour searches are made, so that near-identical frames of its own dataset cannot
hide a copy elsewhere: the --k nearest images in the other datasets, in the other split of its own dataset,
and in its own split of its own dataset. Pairs inside one split of one dataset are mostly consecutive frames of
a video or a fixed camera (0.99-0.997 when fish have moved a little), so they are not duplicates (unless the
files are identical): they stay in the dataset, and the ones from --near-identical-threshold up (default 0.995:
frames repeated by the video, or with fish moved by a few pixels) are listed in near_identical_frames.csv, so
that training can subsample them. Images are counted by their best match in each other dataset, and duplicate
groups only link images that are each other's best match, so a copy of one video frame does not pull in all the
neighbouring frames of its original.

Run on a GPU machine for the full dataset:
    pip install -r requirements.txt "torch>=2.1"
    python -m quality_checks.deduplicate
    python -m quality_checks.deduplicate --datasets ruod_r uiis10k --review 24
    python -m quality_checks.deduplicate --exclude fathomnet deepfish
Descriptors are cached per dataset split in <out-dir>/descriptors, so a rerun only processes new or changed images.

Outputs in <out-dir> (default data/quality/dedup):
    pairs.csv       every pair with similarity >= --min-similarity (>= --threshold inside one split of one
                    dataset), with datasets, splits, box counts and whether the two are each other's best match
    clusters.json   duplicate groups (cross-dataset and train-val mutual best matches, identical files)
    near_identical_frames.csv
                    pairs of near-identical images inside one split of one dataset (kept; to subsample in training)
    report.md       images with a match in each other dataset and in the other split, by similarity band,
                    near-identical frames per dataset, and how many images "keep one image per group" would drop
    review/         side-by-side sample pairs per similarity band (with --review N)
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings

MODEL_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
INPUT_SIZE = 320  # square, as in SSCD's batched "skew_320" preprocessing
DESCRIPTOR_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
BANDS = [0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0]  # similarity bands for the report and the review sheets
KINDS = ["cross-dataset", "same dataset, train-val", "same dataset, same split"]

settings = Settings()


class Images(torch.utils.data.Dataset):
    """Reads each image once: SHA-1 of the file and the normalized 320x320 RGB tensor."""

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        data = self.paths[i].read_bytes()
        try:
            image = Image.open(io.BytesIO(data))
            image.draft("RGB", (INPUT_SIZE, INPUT_SIZE))  # faster JPEG decoding at a reduced scale
            image = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            array = (np.asarray(image, np.float32) / 255 - MEAN) / STD
            ok = True
        except Exception:
            array, ok = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.float32), False
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), hashlib.sha1(data).hexdigest(), ok


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: Path, device: torch.device):
    path = model_dir / Path(MODEL_URL).name
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {MODEL_URL}")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.rename(path)
    return torch.jit.load(str(path), map_location="cpu").to(device).eval()


def list_image_sets(final_dir: Path, only: set[str] | None, exclude: set[str] = frozenset()) -> list[dict]:
    """One entry per <dataset>_train / <dataset>_val folder, sorted by dataset and split, with box counts."""
    sets = []
    for folder in final_dir.iterdir():
        if not (folder / settings.coco_file_name).exists():
            continue
        for split, suffix in (("train", settings.train_dataset_suffix), ("val", settings.val_dataset_suffix)):
            if folder.name.endswith(suffix):
                dataset = folder.name[: -len(suffix)]
                break
        else:
            continue
        if (only and dataset not in only) or dataset in exclude:
            continue
        coco = json.loads((folder / settings.coco_file_name).read_text())
        boxes = Counter(annotation["image_id"] for annotation in coco["annotations"])
        images = [(Path(image["file_name"]).name, boxes[image["id"]]) for image in coco["images"]]
        sets.append({"folder": folder, "dataset": dataset, "split": split, "images": images})
    return sorted(sets, key=lambda s: (s["dataset"], s["split"]))


def describe(image_set: dict, model, device: torch.device, cache_dir: Path, batch_size: int, workers: int):
    """SHA-1 and SSCD descriptor of every image of a set, reusing cached ones for unchanged files."""
    paths = [image_set["folder"] / settings.images_folder_name / name for name, _ in image_set["images"]]
    keys = [f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths]
    cache_path = cache_dir / f"{image_set['folder'].name}.npz"
    cached = {}
    if cache_path.exists():
        with np.load(cache_path) as data:
            cached = {k: i for i, k in enumerate(data["keys"])}
            cached_sha1, cached_desc = data["sha1"], data["descriptors"]
    sha1 = np.array([cached_sha1[cached[k]] if k in cached else "" for k in keys], dtype=object)
    descriptors = np.zeros((len(keys), DESCRIPTOR_SIZE), np.float16)
    for i, k in enumerate(keys):
        if k in cached:
            descriptors[i] = cached_desc[cached[k]]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    unreadable = []
    if todo:
        loader = torch.utils.data.DataLoader(Images([paths[i] for i in todo]), batch_size=batch_size,
                                             num_workers=workers, pin_memory=device.type == "cuda")
        autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
        position = 0
        with torch.no_grad(), autocast:
            for batch, hashes, ok in tqdm(loader, desc=image_set["folder"].name, unit="batch", mininterval=30):
                out = torch.nn.functional.normalize(model(batch.to(device, non_blocking=True)).float(), dim=1)
                for j, row in enumerate(out.cpu().numpy()):
                    i = todo[position + j]
                    sha1[i], descriptors[i] = hashes[j], row
                    if not ok[j]:
                        unreadable.append(paths[i].name)
                        descriptors[i] = 0  # an unreadable image matches nothing
                position += len(batch)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(keys), sha1=sha1.astype(str), descriptors=descriptors)
    return sha1.astype(str), descriptors, unreadable


def top_in_columns(sims: torch.Tensor, ranges: list[tuple[int, int]], k: int):
    """Top-k similarities and column indices of each row within the given column ranges."""
    parts = [(sims[:, a:b].topk(min(k, b - a), dim=1), a) for a, b in ranges if b > a]
    if not parts:
        return None
    values = torch.cat([v for (v, _), _ in parts], 1)
    indices = torch.cat([i + a for (_, i), a in parts], 1)
    values, order = values.topk(min(k, values.shape[1]), dim=1)
    return values, indices.gather(1, order)


def neighbour_pairs(descriptors: np.ndarray, sets: list[tuple[str, str, int, int]], k: int, min_similarity: float,
                    threshold: float, device: torch.device, block_elements: int):
    """
    (i, j, similarity) with i < j for the k nearest neighbours of every image among the other datasets, the
    other split of its dataset, and its own split (from threshold up). sets: (dataset, split, first, end) rows.
    """
    n = len(descriptors)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    all_desc = torch.from_numpy(descriptors).to(device, dtype)
    span = {}
    for dataset, _, lo, hi in sets:
        span[dataset] = (min(lo, span.get(dataset, (lo, hi))[0]), max(hi, span.get(dataset, (lo, hi))[1]))
    rows_per_block = max(1, block_elements // n)
    found_idx, found_sim = [], []
    progress = tqdm(total=n, desc="Step 2/2: neighbour search", unit="img", mininterval=30)
    for dataset, split, lo, hi in sets:
        ds_lo, ds_hi = span[dataset]
        other_split = [(a, b) for d, s, a, b in sets if d == dataset and s != split]
        searches = [([(0, ds_lo), (ds_hi, n)], k, min_similarity), (other_split, k, min_similarity),
                    ([(lo, hi)], k + 1, threshold)]
        for q in range(lo, hi, rows_per_block):
            q_hi = min(hi, q + rows_per_block)
            progress.update(q_hi - q)
            sims = all_desc[q:q_hi] @ all_desc.T
            rows = torch.arange(q, q_hi, device=device)[:, None]
            for ranges, kk, floor in searches:
                top = top_in_columns(sims, ranges, kk)
                if top is None:
                    continue
                values, indices = top
                keep = (values >= floor) & (indices != rows)
                found_idx.append(torch.stack([rows.expand_as(indices)[keep], indices[keep]], 1).cpu().numpy())
                found_sim.append(values[keep].float().cpu().numpy())
    progress.close()
    if not found_idx:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    idx, sim = np.concatenate(found_idx), np.concatenate(found_sim)
    pairs = pd.DataFrame({"a": idx.min(1), "b": idx.max(1), "similarity": sim})
    pairs = pairs.sort_values("similarity", ascending=False).drop_duplicates(["a", "b"])
    return pairs[["a", "b"]].to_numpy(), pairs["similarity"].to_numpy(np.float32)


def union_find_groups(n: int, pairs: np.ndarray) -> list[list[int]]:
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in np.unique(pairs):
        groups[find(i)].append(int(i))
    return list(groups.values())


def write_review(pairs: pd.DataFrame, images: pd.DataFrame, review_dir: Path, per_band: int):
    """For each similarity band, a sheet of random sample pairs (left / right) to judge the threshold by eye."""
    review_dir.mkdir(parents=True, exist_ok=True)
    pairs = pairs[pairs["mutual"] & (pairs["kind"] != KINDS[2])]
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        band = pairs[(pairs["similarity"] >= lo) & ((pairs["similarity"] < hi) | (hi == BANDS[-1]))]
        if band.empty:
            continue
        tiles = []
        for _, pair in band.sample(min(per_band, len(band)), random_state=settings.random_state).iterrows():
            sides = []
            for side in ("a", "b"):
                image = cv2.imread(str(images.at[pair[f"index_{side}"], "path"]))
                image = cv2.resize(image, (max(1, int(image.shape[1] * 200 / image.shape[0])), 200))[:, :300]
                tile = np.zeros((222, 300, 3), np.uint8)
                tile[:200, : image.shape[1]] = image
                cv2.putText(tile, f"{pair[f'dataset_{side}']} {pair[f'split_{side}']}", (3, 216),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                sides.append(tile)
            row = np.hstack(sides + [np.zeros((222, 12, 3), np.uint8)])
            cv2.putText(row, f"{pair['similarity']:.3f}", (245, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(row)
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(review_dir / f"pairs_{lo:.2f}-{hi:.2f}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def mark_best_matches(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    best_for_a: b is a's most similar image in b's dataset (cross-dataset pairs), in the other split (train-val
    pairs) or in its own split (same-split pairs); best_for_b likewise; mutual: both. A copy of one frame of a
    video then links to its original only, not to every neighbouring frame of it.
    """
    pairs = pairs.copy()
    # every pair seen from both of its images, as each image appears as index_a in some pairs and index_b in others
    directed = pd.concat([
        pd.DataFrame({"pair": pairs.index, "side": side, "image": pairs[f"index_{side}"], "match": pairs[f"index_{other}"],
                      "target": pairs[f"dataset_{other}"] + np.where(pairs["kind"] != KINDS[0],
                                                                     "/" + pairs[f"split_{other}"], ""),
                      "similarity": pairs["similarity"]})
        for side, other in (("a", "b"), ("b", "a"))])
    # exactly one best match per image and target: similarities tie often (half-precision descriptors), and ties
    # would chain consecutive frames into one group
    best = directed.sort_values(["similarity", "match"], ascending=[False, True]).drop_duplicates(["image", "target"])
    for side in ("a", "b"):
        pairs[f"best_for_{side}"] = pairs.index.isin(best.loc[best["side"] == side, "pair"])
    pairs["mutual"] = pairs["best_for_a"] & pairs["best_for_b"]
    return pairs


def report(images: pd.DataFrame, pairs: pd.DataFrame, threshold: float, near_identical_threshold: float
           ) -> tuple[str, list[list[int]]]:
    dup = pairs[pairs["similarity"] >= threshold]
    lines = ["# Duplicate report", "",
             f"{len(images):,} images in {images['dataset'].nunique()} datasets. A duplicate is a pair of images in two "
             f"datasets or in the two splits of one dataset with SSCD cosine similarity >= {threshold}, or identical "
             f"files (SHA-1). Pairs inside one split from {near_identical_threshold} up are near-identical frames: they "
             "stay, and are listed for subsampling in training.", ""]

    # Each image counted once per other dataset (or other split), by its best match there
    directed = pd.concat([
        pd.DataFrame({"image": pairs[f"index_{s}"], "dataset": pairs[f"dataset_{s}"], "split": pairs[f"split_{s}"],
                      "other": pairs[f"dataset_{o}"], "other_split": pairs[f"split_{o}"], "kind": pairs["kind"],
                      "similarity": pairs["similarity"]})
        for s, o in (("a", "b"), ("b", "a"))])
    directed = directed[directed["kind"] != KINDS[2]]
    best = directed.groupby(["image", "dataset", "split", "other", "other_split", "kind"], as_index=False)["similarity"].max()
    bands = [b for b in BANDS[:-1] if b >= 0.7]
    header = " | ".join(f">= {b:.2f}" for b in bands)
    lines += ["## Images with a match in another dataset, by similarity of their best match", "",
              "Each row counts the images of the first dataset whose most similar image in the second one reaches the "
              "similarity. Neighbouring frames of a video also match at 0.75-0.9, identical photos mostly above 0.9.", "",
              f"| dataset | split | matched in | {header} |", "|---|---|---|" + "---|" * len(bands)]
    cross = best[best["kind"] == KINDS[0]]
    for (dataset, split, other), g in cross.groupby(["dataset", "split", "other"]):
        if g["similarity"].max() >= bands[0]:
            counts = " | ".join(f"{int((g['similarity'] >= b).sum()):,}" for b in bands)
            lines.append(f"| {dataset} | {split} | {other} ({', '.join(sorted(g['other_split'].unique()))}) | {counts} |")
    lines.append("")

    lines += ["## Val images with a match in the train split of their own dataset", "",
              f"| dataset | val images | {header} |", "|---|---|" + "---|" * len(bands)]
    inside = best[(best["kind"] == KINDS[1]) & (best["split"] == "val")]
    val_counts = images[images["split"] == "val"].groupby("dataset").size()
    for dataset, g in inside.groupby("dataset"):
        if g["similarity"].max() >= bands[0]:
            counts = " | ".join(f"{int((g['similarity'] >= b).sum()):,}" for b in bands)
            lines.append(f"| {dataset} | {val_counts.get(dataset, 0):,} | {counts} |")
    lines += ["", "In datasets from fixed cameras or videos, these are mostly frames of the same scene, not copies.", ""]

    same = dup[dup["kind"] == KINDS[2]]
    lines += ["## Near-identical frames inside one split of a dataset (kept, listed in near_identical_frames.csv)", "",
              f"Pairs from {near_identical_threshold} up are frames repeated by a video or with fish moved by a few "
              f"pixels; below it (from {threshold}) they are mostly consecutive frames of a video or a fixed camera.", "",
              f"| dataset | split | pairs >= {threshold} | pairs >= {near_identical_threshold} | images in them | "
              "identical files |", "|---|---|---|---|---|---|"]
    for (dataset, split), sel in same.groupby(["dataset_a", "split_a"]):
        near = sel[sel["similarity"] >= near_identical_threshold]
        involved = len(np.unique(near[["index_a", "index_b"]].to_numpy()))
        lines.append(f"| {dataset} | {split} | {len(sel):,} | {len(near):,} | {involved:,} | "
                     f"{int(sel['identical_file'].sum()):,} |")
    lines.append("")

    # Keep one image per group (a val image if any, then the one with most boxes): how many would be dropped
    links = dup[(dup["mutual"] & (dup["kind"] != KINDS[2])) | dup["identical_file"]]
    groups = union_find_groups(len(images), links[["index_a", "index_b"]].to_numpy())
    dropped = []
    for group in groups:
        keep = max(group, key=lambda i: (images.at[i, "split"] == "val", images.at[i, "boxes"], -i))
        dropped += [i for i in group if i != keep]
    drops = images.loc[dropped].groupby(["dataset", "split"]).size()
    lines += ["## If one image per duplicate group were kept", "",
              f"{len(groups):,} groups, linked by cross-dataset and train-val pairs at >= {threshold} where each image is "
              "the other's best match, and by identical files. Keeping one image per group (a val image if there is one, "
              "else the one with most boxes) would drop:", "",
              "| dataset | train images dropped | val images dropped | share of its images |", "|---|---|---|---|"]
    totals = images.groupby("dataset").size()
    for dataset in sorted({d for d, _ in drops.index}):
        train, val = int(drops.get((dataset, "train"), 0)), int(drops.get((dataset, "val"), 0))
        lines.append(f"| {dataset} | {train:,} | {val:,} | {(train + val) / totals[dataset]:.1%} |")
    lines += ["", f"Total: {len(dropped):,} of {len(images):,} images."]
    return "\n".join(lines) + "\n", groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-dir", type=Path, default=settings.processed_dir)
    parser.add_argument("--out-dir", type=Path, default=settings.base_dir / "quality" / "dedup")
    parser.add_argument("--datasets", nargs="*", help="only these dataset shortnames (default: all)")
    parser.add_argument("--exclude", nargs="*", default=[], help="skip these dataset shortnames")
    parser.add_argument("--threshold", type=float, default=0.9, help="SSCD similarity of a duplicate")
    parser.add_argument("--near-identical-threshold", type=float, default=0.995,
                        help="SSCD similarity from which two images inside one split are listed as near-identical")
    parser.add_argument("--min-similarity", type=float, default=0.6,
                        help="lowest similarity of cross-dataset and train-val pairs written to pairs.csv")
    parser.add_argument("--k", type=int, default=10, help="neighbours kept per image in each search")
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--block-elements", type=int, default=2**28,
                        help="elements of one block of the similarity matrix (query rows x images)")
    parser.add_argument("--review", type=int, default=0, help="sample pairs per similarity band for review sheets")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}")
    model = load_model(args.out_dir / "models", device)
    print("Reading COCO annotations...", flush=True)
    image_sets = list_image_sets(args.final_dir, set(args.datasets) if args.datasets else None, set(args.exclude))
    rows, sha1s, descriptors, spans = [], [], [], []
    total = sum(len(s["images"]) for s in image_sets)
    print(f"{total:,} images in {len(image_sets)} dataset splits", flush=True)
    for n_set, image_set in enumerate(image_sets, 1):
        print(f"Step 1/2: descriptors [{n_set}/{len(image_sets)}] {image_set['folder'].name}: "
              f"{len(image_set['images']):,} images ({len(rows):,}/{total:,} = {len(rows) / total:.1%} done)", flush=True)
        sha1, desc, unreadable = describe(image_set, model, device, args.out_dir / "descriptors", args.batch_size,
                                          args.workers)
        if unreadable:
            print(f"WARNING: {len(unreadable)} unreadable images in {image_set['folder'].name}: {unreadable[:5]}")
        spans.append((image_set["dataset"], image_set["split"], len(rows), len(rows) + len(sha1)))
        for name, boxes in image_set["images"]:
            rows.append({"dataset": image_set["dataset"], "split": image_set["split"], "file_name": name,
                         "boxes": boxes, "path": image_set["folder"] / settings.images_folder_name / name})
        sha1s.append(sha1)
        descriptors.append(desc)
    images = pd.DataFrame(rows)
    images["sha1"] = np.concatenate(sha1s)
    descriptors = np.concatenate(descriptors).astype(np.float32)
    print(f"{len(images):,} images in {len(image_sets)} dataset splits")

    pair_idx, sims = neighbour_pairs(descriptors, spans, args.k, args.min_similarity, args.threshold, device,
                                     args.block_elements)
    # identical files are duplicates whatever their neighbour rank
    exact = np.array([(g[0], j) for g in images.groupby("sha1").indices.values() if len(g) > 1 for j in g[1:]],
                     np.int64).reshape(-1, 2)
    pairs = pd.concat([pd.DataFrame({"index_a": pair_idx[:, 0], "index_b": pair_idx[:, 1], "similarity": sims}),
                       pd.DataFrame({"index_a": exact[:, 0], "index_b": exact[:, 1], "similarity": 1.0})])
    pairs = pairs.sort_values("similarity", ascending=False).drop_duplicates(["index_a", "index_b"])
    for side in ("a", "b"):
        for column in ("dataset", "split", "file_name", "boxes", "sha1"):
            pairs[f"{column}_{side}"] = images[column].to_numpy()[pairs[f"index_{side}"].to_numpy()]
    pairs["identical_file"] = pairs["sha1_a"] == pairs["sha1_b"]
    pairs.loc[pairs["identical_file"], "similarity"] = 1.0
    pairs["kind"] = np.where(pairs["dataset_a"] != pairs["dataset_b"], KINDS[0],
                             np.where(pairs["split_a"] != pairs["split_b"], KINDS[1], KINDS[2]))
    pairs = mark_best_matches(pairs.sort_values("similarity", ascending=False).reset_index(drop=True))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["similarity", "identical_file", "kind", "mutual"] + [f"{c}_{s}" for s in ("a", "b")
                                                                    for c in ("dataset", "split", "file_name", "boxes")]
    pairs[columns].to_csv(args.out_dir / "pairs.csv", index=False, float_format="%.4f")
    near = pairs[(pairs["kind"] == KINDS[2]) & (pairs["similarity"] >= args.near_identical_threshold)
                 & ~pairs["identical_file"]]
    near_columns = {"similarity": "similarity", "dataset_a": "dataset", "split_a": "split", "file_name_a": "file_a",
                    "file_name_b": "file_b", "boxes_a": "boxes_a", "boxes_b": "boxes_b"}
    near[list(near_columns)].rename(columns=near_columns).to_csv(args.out_dir / "near_identical_frames.csv", index=False,
                                                                 float_format="%.4f")
    text, groups = report(images, pairs, args.threshold, args.near_identical_threshold)
    (args.out_dir / "report.md").write_text(text)
    with open(args.out_dir / "clusters.json", "w") as f:
        names = (images["dataset"] + "_" + images["split"] + "/" + images["file_name"]).to_numpy()
        json.dump([[names[i] for i in g] for g in sorted(groups, key=len, reverse=True)], f, indent=1)
    if args.review:
        write_review(pairs, images, args.out_dir / "review", args.review)
    print(text)
    print(f"Outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
