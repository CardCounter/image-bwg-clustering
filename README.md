# Image BWG Clustering

Utility script to cluster images into binary or trinary black/white/grey (BWG)
representations, or into quad-tone ASCII art. Images placed in the `input/`
directory are clustered with a pixel based renderer, a stylised circular
representation, or exported as ASCII art and saved to the `output/` directory.
By default, pixel mode exports 512×512 PNG files, circle mode clusters on a
24×24 grid and renders crisp circular dots that are upscaled to 512×512, and
ASCII mode produces a 64×64-character text file.

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

3. Clustered outputs are written to `output/` using the naming pattern
   `<original-name>_<mode>_<clusters>.<ext>`, where `.png` is used for the
   raster modes and `.txt` for ASCII art.

## Command Line Options

| Option          | Description                                                     | Default |
|-----------------|-----------------------------------------------------------------|---------|
| `--input-dir`   | Directory containing images to process.                         | `input` |
| `--output-dir`  | Destination directory for clustered PNG files.                  | `output`|
| `--mode`        | Rendering style: `pixel`, `circle`, or `ascii`.                   | `pixel` |
| `--clusters`    | Number of clusters for image outputs (`2` or `3`). ASCII mode always renders four tone levels. | `2`     |
| `--size`        | Output size; 512 for images and 64 for ASCII art if unspecified. | –       |
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
- **ASCII mode**: converts each clustered pixel into characters (`@`, `#`, `:`,
  or space) and writes the result as a plain text file using four tonal
  clusters. The default output grid is 64×64 characters; specify `--size` to
  change this resolution.
