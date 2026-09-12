# Digital Image Processing Experiment

The uploaded nature image is used as the source image.

## Processing
The colour image is resized to **600 × 400** for a practical-size lab experiment and converted to grayscale:
`Gray = 0.299R + 0.587G + 0.114B`

## Operations
- Haar 2-D DWT: 1-level and 2-level multiresolution
- Haar IDWT reconstruction
- Shannon-Fano source coding
- Huffman source coding
- First-order edge detection: Forward, Backward, Central Difference
- Second-order edge detection: LoG, DoG and Zero-Crossing

## Measured results for the supplied image
- DWT/IDWT MSE: 1.0173e-27
- Maximum reconstruction error: 1.7053e-13
- Entropy: 7.653274 bits/pixel
- Shannon-Fano average length: 7.705958 bits/symbol
- Shannon-Fano efficiency: 99.32%
- Huffman average length: 7.682900 bits/symbol
- Huffman efficiency: 99.61%

## Run
```bash
pip install -r requirements.txt
python main.py
```

Open the `output` folder to see all generated result images.
