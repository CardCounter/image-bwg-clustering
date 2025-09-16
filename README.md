# Image BWG Clustering

Utility script to cluster images into binary or trinary black/white/grey (BWG)
representations. Images placed in the `input/` directory are clustered with
either a pixel based renderer or a stylised circular representation and saved
to the `output/` directory. By default, pixel mode exports 512×512 PNG files,
while circle mode clusters on a 24×24 grid and renders crisp circular dots that
are upscaled to 512×512 for easier viewing.

## Requirements

The script relies on [Pillow](https://python-pillow.org/) and
[NumPy](https://numpy.org/). Both packages are commonly available in Python
scientific distributions. To install them manually run:

```bash
pip install pillow numpy
```

## Usage

1. Place one or more images (PNG, JPG/JPEG, BMP, GIF, TIFF, WEBP) inside the
   `input/` directory.
2. Run the script:

   ```bash
   python main.py --mode pixel --clusters 2
   ```

3. Clustered PNG files are written to `output/` using the naming pattern
   `<original-name>_<mode>_<clusters>.png`.

## Command Line Options

| Option          | Description                                                     | Default |
|-----------------|-----------------------------------------------------------------|---------|
| `--input-dir`   | Directory containing images to process.                         | `input` |
| `--output-dir`  | Destination directory for clustered PNG files.                  | `output`|
| `--mode`        | Rendering style, either `pixel` or `circle`.                     | `pixel` |
| `--clusters`    | Number of clusters (`2` for binary, `3` for trinary).            | `2`     |
| `--size`        | Output size; images are resized to a square of this dimension.   | `512`   |
| `--max-iter`    | Maximum iterations used by the k-means clustering algorithm.     | `30`    |
| `--tol`         | Early stopping tolerance for k-means centroid movement.          | `1e-4`  |
| `--seed`        | Random seed for centroid initialisation.                         | `42`    |

### Rendering Modes

- **Pixel mode**: produces a rasterised image. Binary clustering outputs a black
  and white image, while trinary clustering outputs black, grey, and white
  clusters.
- **Circle mode**: renders each clustered pixel as a circular dot on a white
  background, producing a stippled look while retaining the original layout.
  With three clusters, the mid-intensity group uses smaller black dots instead
  of grey ones. The clustering grid is fixed at 24×24, and the rendered dots are
  upscaled to a 512×512 output unless you override `--size`.
