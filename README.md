# Animal Classification - Image Preprocessing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Visit: https://www.kaggle.com/datasets/artemgoncarov/animals-segmentation-and-detection
   - Extract to `animal_data/` folder

   Expected structure:
   ```
   animal_data/
   ├── train/images/
   ├── val/images/
   └── test/images/
   ```

## Usage

```bash
python3 image_preprocess.py
```

## What It Does

- Resizes images to 416x416 with aspect ratio preservation
- Normalizes pixel values (0-255 → 0.0-1.0)
- Randomly flips 10% of images for data augmentation
- Saves processed images to `processed_images/` by split

## Configuration

Edit `image_preprocess.py` to adjust:
- `TARGET_SIZE` - Image dimensions (default: 416x416)
- `FLIP_PROBABILITY` - Flip percentage (default: 0.1)