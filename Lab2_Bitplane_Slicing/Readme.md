# Bit Plane Slicing

This lab breaks a grayscale image down into its 8 individual bit planes.

## What is bit plane slicing?

Every pixel in an 8-bit grayscale image is stored as an 8-bit binary
number (bits 0 to 7). Bit plane slicing separates the image into 8
separate binary images — one for each bit position:

- **Bit 7 (MSB)** – carries the most visual information, looks closest
  to the original image.
- **Bit 0 (LSB)** – carries the least visual information, looks like
  random noise.

For each bit position `k`, the plane is extracted as:

```
plane_k = (pixel_value >> k) & 1
```

then scaled to 0/255 so it displays as a black & white image.

## Files

- `bit_plane_slicing.py` – Extracts and displays all 8 bit planes
- `nature.jpg` – Sample input image
- `bit_plane_slicing_output.png` – Output showing the original image and all 8 bit planes

## How to run

```bash
pip install opencv-python numpy matplotlib

python bit_plane_slicing.py nature.jpg
```

This saves `bit_plane_slicing_output.png`, showing the original grayscale
image alongside Bit Plane 7 down to Bit Plane 0.
