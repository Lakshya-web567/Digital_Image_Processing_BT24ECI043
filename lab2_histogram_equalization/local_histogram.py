import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def equalize_block(block: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
    cdf = np.cumsum(hist)

    cdf_nonzero = cdf[cdf > 0]
    if cdf_nonzero.size == 0:
        return block  # flat/empty block, nothing to equalize

    cdf_min = cdf_nonzero.min()
    total = block.size
    denom = max(total - cdf_min, 1)  # avoid divide-by-zero on flat blocks

    lut = np.round((cdf - cdf_min) / denom * 255).clip(0, 255).astype(np.uint8)
    return lut[block]


def local_histogram_equalization(gray_img: np.ndarray, block_size: int = 32) -> np.ndarray:
    h, w = gray_img.shape
    output = np.zeros_like(gray_img)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray_img[y:y + block_size, x:x + block_size]
            output[y:y + block_size, x:x + block_size] = equalize_block(block)

    return output


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "monument.jpg"
    block_size = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    local_eq = local_histogram_equalization(gray, block_size)

    hist_orig, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_eq, _ = np.histogram(local_eq.ravel(), bins=256, range=(0, 256))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(local_eq, cmap="gray")
    axes[0, 1].set_title(f"Local Histogram Equalized (block={block_size})")
    axes[0, 1].axis("off")

    axes[1, 0].plot(hist_orig, color="black")
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].plot(hist_eq, color="black")
    axes[1, 1].set_title("Local Equalized Histogram")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("local_histogram_output.png", dpi=150)
    print("Saved -> local_histogram_output.png")


if __name__ == "__main__":
    main()
