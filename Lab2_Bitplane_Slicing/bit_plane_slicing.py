import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_bit_plane(gray_img: np.ndarray, bit: int) -> np.ndarray:
    
    plane = (gray_img >> bit) & 1
    return plane * 255

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "nature.jpg"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract all 8 bit planes, from bit 7 (MSB) down to bit 0 (LSB)
    planes = [get_bit_plane(gray, bit) for bit in range(7, -1, -1)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original (Grayscale) Image")
    axes[0].axis("off")

    for i, (bit, plane) in enumerate(zip(range(7, -1, -1), planes)):
        axes[i + 1].imshow(plane, cmap="gray")
        axes[i + 1].set_title(f"Bit Plane {bit}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("bit_plane_slicing_output.png", dpi=150)
    print("Saved -> bit_plane_slicing_output.png")

if __name__ == "__main__":
    main()
