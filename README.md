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
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       └── images/
   ```

## Usage

```bash
python3 image_preprocess.py
```

## What It Does

- Resizes images to 416x416 with aspect ratio preservation and padding
- Normalizes pixel values (0-255 → 0.0-1.0)
- Applies brightness augmentation to 10% of training images (preserves bounding boxes)
- Copies corresponding YOLO label files to maintain image-label correspondence
- Saves processed images and labels to `processed_images/` organized by split

## Output Structure

```
processed_images/
├── train/
│   ├── images/           # Processed training images
│   └── labels/           # Corresponding YOLO label files
├── val/
│   ├── images/           # Processed validation images  
│   └── labels/           # Corresponding YOLO label files
└── test/
    └── images/           # Processed test images (no labels)
```

## Configuration

Edit `image_preprocess.py` to adjust:
- `TARGET_SIZE` - Image dimensions (default: 416x416)
- `AUGMENT_PROBABILITY` - Brightness augmentation percentage (default: 0.1)

## Features

- **Label preservation**: Automatically copies and renames label files to match processed images
- **Smart augmentation**: Only brightness adjustment (no geometric changes) to preserve bounding box coordinates
- **Skip processing**: Automatically skips already processed directories
- **Progress tracking**: Shows progress bars during processing