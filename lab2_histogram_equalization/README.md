# Lab 2: Histogram Equalization

This lab compares three ways of doing histogram equalization on an image:
**Global**, **Local**, and **Adaptive**.

## Files

- `global_histogram.py` – Global histogram equalization
- `local_histogram.py` – Local (block-based) histogram equalization
- `adaptive_histogram.py` – Adaptive histogram equalization (CLAHE-style)
- `GUI.py` – Simple GUI to load an image and try all three methods
- `nature.jpg` – Sample input image
- `global_histogram_output.png` – Output of global method
- `local_histogram_output.png` – Output of local method
- `adaptive_histogram_output.png` – Output of adaptive method

## What each method does

**Global HE** – Looks at the whole image's histogram and stretches it out to
use the full range of brightness values. Simple and fast, but can wash out
detail in some areas.

**Local HE** – Splits the image into small blocks and equalizes each block
on its own. Brings out more local detail, but creates visible blocky
edges between tiles.

**Adaptive HE** – Same idea as local HE, but smooths out the blockiness
using interpolation between tiles, and limits contrast in flat areas so
noise doesn't get amplified. Gives the best-looking result overall.

## How to run

```bash
pip install opencv-python numpy matplotlib

python global_histogram.py nature.jpg
python local_histogram.py nature.jpg
python adaptive_histogram.py nature.jpg

# or open the GUI
python GUI.py
```

Each script saves an output image comparing the original and equalized
image, along with their histograms.
