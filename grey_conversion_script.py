import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load the image
img = cv2.imread("input.jpg")

if img is None:
    raise FileNotFoundError("input.jpg not found. Place an image named 'input.jpg' in this folder.")

# OpenCV loads images as BGR, convert to RGB for correct display in Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- Convert to grayscale ----
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- Split into B, G, R channels ----
b, g, r = cv2.split(img)
zeros = np.zeros_like(b)

# ---- Reconstruct single-channel "true color" images ----
red_channel = cv2.merge([zeros, zeros, r])     # only Red channel active
green_channel = cv2.merge([zeros, g, zeros])   # only Green channel active
blue_channel = cv2.merge([b, zeros, zeros])    # only Blue channel active

# Convert reconstructed channel images (still BGR order) to RGB for display
red_channel_rgb = cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB)
green_channel_rgb = cv2.cvtColor(green_channel, cv2.COLOR_BGR2RGB)
blue_channel_rgb = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2RGB)

# Display all 5 images in a single figure 
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original")

axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale")

axes[0, 2].imshow(red_channel_rgb)
axes[0, 2].set_title("Red Channel")

axes[1, 0].imshow(green_channel_rgb)
axes[1, 0].set_title("Green Channel")

axes[1, 1].imshow(blue_channel_rgb)
axes[1, 1].set_title("Blue Channel")

axes[1, 2].axis("off")  # empty slot

for ax in axes.ravel():
    ax.axis("off") if ax != axes[1, 2] else None
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("output.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done. Saved result as output.png")
