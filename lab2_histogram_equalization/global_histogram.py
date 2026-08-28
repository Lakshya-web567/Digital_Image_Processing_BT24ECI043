import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_histogram(gray_img: np.ndarray) -> np.ndarray:
    return np.bincount(gray_img.ravel(), minlength=256)


def global_histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    hist = compute_histogram(gray_img)
    cdf = np.cumsum(hist)

    # Normalize CDF to [0, 255]. Mask out the zero entries at the start
    # (cdf_min) so pure-black/white images don't get divided by zero.
    cdf_min = cdf[cdf > 0].min()
    total_pixels = gray_img.size

    lut = np.round(
        (cdf - cdf_min) / (total_pixels - cdf_min) * 255
    ).clip(0, 255).astype(np.uint8)

    equalized = lut[gray_img]
    return equalized


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "monument.jpg"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = global_histogram_equalization(gray)

    hist_orig = compute_histogram(gray)
    hist_eq = compute_histogram(equalized)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(equalized, cmap="gray")
    axes[0, 1].set_title("Global Histogram Equalized")
    axes[0, 1].axis("off")

    axes[1, 0].plot(hist_orig, color="black")
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].plot(hist_eq, color="black")
    axes[1, 1].set_title("Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("global_histogram_output.png", dpi=150)
    print("Saved -> global_histogram_output.png")

if __name__ == "__main__":
    main()
