# Animal Classification - Image Preprocessing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset:**
   - Visit: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset/data
   - Extract to `animal_data/` folder

   Expected structure:
   ```
   animal_data/
   ├── train/
   │   ├── Bear/
   │   │   ├── Label/          # Label files (.txt format)
   │   │   └── *.jpg           # Bear images
   │   ├── Brown bear/
   │   │   ├── Label/
   │   │   └── *.jpg
   │   └── ... (70+ animal classes)
   └── test/
       ├── Bear/
       │   ├── Label/
       │   └── *.jpg
       └── ... (same classes)
   ```

## Usage

```bash
python3 image_preprocess.py
```

## What It Does

- **Dual Format Processing**: Converts labels to both YOLO format (for object detection) and preserves original format (for CNN classification)
- **Image Preprocessing**: Resizes images to 416x416 with aspect ratio preservation and padding
- **Normalization**: Converts pixel values from 0-255 to 0.0-1.0 range
- **Data Augmentation**: Applies brightness and horizontal flip augmentation to 10% of training images
- **Label Conversion**: Converts absolute coordinates to normalized YOLO format (class_id x_center y_center width height)
- **Class Mapping**: Automatically generates class ID mappings from directory structure

## Label Format

**Input format** (from dataset):
```
Bear 212.41856 134.383104 741.982208 627.37536
```

**Output formats**:
- **YOLO**: `0 0.512 0.324 0.356 0.487` (normalized coordinates)
- **Original**: `Bear 212.41856 134.383104 741.982208 627.37536` (preserved)

## Output Structure

```
processed_images/
├── train/
│   ├── images/              # Processed training images (416x416)
│   ├── labels_yolo/         # YOLO format labels (normalized coordinates)
│   ├── labels_original/     # Original format labels (absolute coordinates)
│   ├── classes.txt          # Class ID to name mapping
│   └── yolo_config.yaml     # YOLO dataset configuration
└── test/
    ├── images/              # Processed test images
    ├── labels_yolo/         # YOLO format test labels
    ├── labels_original/     # Original format test labels
    ├── classes.txt          # Class mapping
    └── yolo_config.yaml     # YOLO configuration
```