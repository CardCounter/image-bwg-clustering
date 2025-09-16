#!/usr/bin/env python3
"""Cluster input images into binary or trinary representations.

The script reads every supported image file from an input directory, clusters
its pixels into two or three groups and writes a 512x512 PNG to the output
directory.  Two rendering modes are available:

* ``pixel`` – generates a black/white (binary) or black/grey/white (trinary)
  raster image based on the cluster labels.
* ``circle`` – places solid black circles on a white background whose
  positions and sizes are derived from the cluster statistics.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps


try:  # Pillow changed the resampling namespace in version 10.0
    _RESAMPLE = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - compatibility branch
    _RESAMPLE = Image.LANCZOS  # type: ignore[assignment]


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the clustering script."""

    parser = argparse.ArgumentParser(
        description=(
            "Cluster images into binary or trinary representations and save the "
            "results as 512x512 PNG files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing the images to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where the clustered PNG files are written.",
    )
    parser.add_argument(
        "--mode",
        choices=["pixel", "circle"],
        default="pixel",
        help="Rendering mode: pixel cluster map or stylised circles.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of clusters to compute (binary or trinary).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size (images are resized to SIZE x SIZE).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=30,
        help="Maximum number of k-means iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Early stopping tolerance for the centroid displacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for centroid initialisation.",
    )
    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    """Create *path* (and parents) if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)


def iter_images(directory: Path) -> Iterable[Path]:
    """Yield all supported image files inside *directory*."""

    if not directory.exists():
        return

    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield entry


def load_image(path: Path, size: int) -> Image.Image:
    """Load *path* as an Image resized to ``size x size`` pixels."""

    with Image.open(path) as img:
        prepared = ImageOps.exif_transpose(img)
        fitted = ImageOps.fit(prepared, (size, size), method=_RESAMPLE)
    return fitted


def image_to_data(image: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convert *image* to grayscale data for clustering."""

    grayscale = image.convert("L")
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    height, width = array.shape
    flat = array.reshape(-1, 1)
    return flat, (height, width)


def kmeans(
    data: np.ndarray,
    k: int,
    *,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """A lightweight k-means implementation for small *k* values."""

    if data.shape[0] < k:
        raise ValueError("Not enough data points to form the requested clusters.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(data.shape[0], size=k, replace=False)
    centroids = np.copy(data[indices])

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        new_centroids = np.copy(centroids)
        for idx in range(k):
            members = data[labels == idx]
            if members.size == 0:
                new_centroids[idx] = data[rng.integers(0, data.shape[0])]
            else:
                new_centroids[idx] = members.mean(axis=0)

        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if shift <= tol:
            break

    distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    return centroids, labels


def remap_labels(labels: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder labels so that centroid intensities are monotonic."""

    order = np.argsort(centroids[:, 0])
    lookup = np.zeros_like(order)
    for new_idx, old_idx in enumerate(order):
        lookup[old_idx] = new_idx
    remapped = lookup[labels]
    return remapped, centroids[order]


def render_pixel_clusters(label_grid: np.ndarray, clusters: int) -> Image.Image:
    """Convert a label grid to a grayscale output image."""

    if clusters == 2:
        palette = np.array([0, 255], dtype=np.uint8)
    else:
        palette = np.array([0, 128, 255], dtype=np.uint8)

    raster = palette[label_grid]
    return Image.fromarray(raster, mode="L")


def render_circle_clusters(label_grid: np.ndarray) -> Image.Image:
    """Draw black circles for each cluster on a white canvas."""

    height, width = label_grid.shape
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)

    total = label_grid.size
    cluster_details = []
    for label in np.unique(label_grid):
        coords = np.column_stack(np.where(label_grid == label))
        if coords.size == 0:
            continue
        count = coords.shape[0]
        center_y, center_x = coords.mean(axis=0)
        cluster_details.append((count, (float(center_x), float(center_y))))

    cluster_details.sort(reverse=True, key=lambda item: item[0])

    for count, (center_x, center_y) in cluster_details:
        ratio = count / total
        radius = max(6.0, math.sqrt(ratio) * (min(width, height) / 1.8))
        left = max(center_x - radius, 0.0)
        top = max(center_y - radius, 0.0)
        right = min(center_x + radius, width)
        bottom = min(center_y + radius, height)
        draw.ellipse((left, top, right, bottom), fill=0)

    return canvas


def process_image(
    image_path: Path,
    output_dir: Path,
    *,
    mode: str,
    clusters: int,
    size: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Path:
    """Cluster a single image and write the PNG result."""

    image = load_image(image_path, size)
    data, (height, width) = image_to_data(image)
    centroids, labels = kmeans(data, clusters, max_iter=max_iter, tol=tol, seed=seed)
    remapped_labels, _ = remap_labels(labels, centroids)
    label_grid = remapped_labels.reshape((height, width))

    if mode == "pixel":
        output_image = render_pixel_clusters(label_grid, clusters)
    else:
        output_image = render_circle_clusters(label_grid)

    output_path = output_dir / f"{image_path.stem}_{mode}_{clusters}.png"
    output_image.save(output_path, format="PNG")
    return output_path


def main() -> None:
    args = parse_args()

    ensure_directory(args.input_dir)
    ensure_directory(args.output_dir)

    images = list(iter_images(args.input_dir))
    if not images:
        print(f"No supported images found in {args.input_dir.resolve()}")
        return

    for image_path in images:
        result = process_image(
            image_path,
            args.output_dir,
            mode=args.mode,
            clusters=args.clusters,
            size=args.size,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
        )
        print(f"Wrote {result.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
