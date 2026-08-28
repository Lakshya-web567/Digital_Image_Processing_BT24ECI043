import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def build_tile_lut(tile: np.ndarray, clip_limit_factor: float) -> np.ndarray:
    hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 256))

    # Contrast limiting
    n_pixels = tile.size
    avg_bin_height = n_pixels / 256.0
    clip_limit = max(int(avg_bin_height * clip_limit_factor), 1)

    excess = np.sum(np.maximum(hist - clip_limit, 0))
    hist = np.minimum(hist, clip_limit)
    # redistribute clipped mass evenly across all 256 bins
    hist = hist + (excess / 256.0)

    # Build LUT from clipped CDF 
    cdf = np.cumsum(hist)
    cdf_min = cdf.min()
    denom = max(cdf[-1] - cdf_min, 1e-6)
    lut = np.round((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.float32)
    return lut


def adaptive_histogram_equalization(
    gray_img: np.ndarray, tiles: int = 8, clip_limit_factor: float = 2.0
) -> np.ndarray:
    h, w = gray_img.shape
    tile_h = h // tiles
    tile_w = w // tiles

    # Guard against tiny images / too many tiles
    tile_h = max(tile_h, 1)
    tile_w = max(tile_w, 1)
    n_tiles_y = int(np.ceil(h / tile_h))
    n_tiles_x = int(np.ceil(w / tile_w))

    # 1. Build one LUT per tile
    luts = np.zeros((n_tiles_y, n_tiles_x, 256), dtype=np.float32)
    centers_y = np.zeros(n_tiles_y)
    centers_x = np.zeros(n_tiles_x)

    for ty in range(n_tiles_y):
        y0, y1 = ty * tile_h, min((ty + 1) * tile_h, h)
        centers_y[ty] = (y0 + y1) / 2.0
        for tx in range(n_tiles_x):
            x0, x1 = tx * tile_w, min((tx + 1) * tile_w, w)
            centers_x[tx] = (x0 + x1) / 2.0
            tile = gray_img[y0:y1, x0:x1]
            luts[ty, tx] = build_tile_lut(tile, clip_limit_factor)

    ys = np.arange(h)
    xs = np.arange(w)

    if n_tiles_y > 1:
        ty0 = np.clip(np.searchsorted(centers_y, ys) - 1, 0, n_tiles_y - 2)
        ty1 = ty0 + 1
        wy = np.clip((ys - centers_y[ty0]) / (centers_y[ty1] - centers_y[ty0]), 0, 1)
    else:
        ty0 = ty1 = np.zeros(h, dtype=int)
        wy = np.zeros(h)

    if n_tiles_x > 1:
        tx0 = np.clip(np.searchsorted(centers_x, xs) - 1, 0, n_tiles_x - 2)
        tx1 = tx0 + 1
        wx = np.clip((xs - centers_x[tx0]) / (centers_x[tx1] - centers_x[tx0]), 0, 1)
    else:
        tx0 = tx1 = np.zeros(w, dtype=int)
        wx = np.zeros(w)

    # Reshape row-wise values to columns and column-wise values to rows
    # so they broadcast together into full (h, w) grids.
    TY0, TX0 = ty0[:, None], tx0[None, :]
    TY1, TX1 = ty1[:, None], tx1[None, :]
    WY, WX = wy[:, None], wx[None, :]

    # For every pixel, look up its value's mapped result in each of the
    # 4 surrounding tile LUTs, then blend by distance (bilinear weights).
    v00 = luts[TY0, TX0, gray_img]
    v01 = luts[TY0, TX1, gray_img]
    v10 = luts[TY1, TX0, gray_img]
    v11 = luts[TY1, TX1, gray_img]

    top = v00 * (1 - WX) + v01 * WX
    bottom = v10 * (1 - WX) + v11 * WX
    output = top * (1 - WY) + bottom * WY

    return output.clip(0, 255).astype(np.uint8)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "monument.jpg"
    tiles = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    clip_limit_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_eq = adaptive_histogram_equalization(gray, tiles, clip_limit_factor)

    hist_orig, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_eq, _ = np.histogram(adaptive_eq.ravel(), bins=256, range=(0, 256))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(adaptive_eq, cmap="gray")
    axes[0, 1].set_title("Adaptive Histogram Equalized")
    axes[0, 1].axis("off")

    axes[1, 0].plot(hist_orig, color="black")
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].plot(hist_eq, color="black")
    axes[1, 1].set_title("Adaptive Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("adaptive_histogram_output.png", dpi=150)
    print("Saved -> adaptive_histogram_output.png")

if __name__ == "__main__":
    main()
