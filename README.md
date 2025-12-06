# Animal Classification - Image Preprocessing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset:**
   - Put the dataset `Animals-10/` folder in the same directory as the main.py file
   - Dataset from: https://www.kaggle.com/datasets/viratkothari/animal10 
    
   Expected structure:
   ```
   Animals-10/
   ├── train/
   │   ├── butterfly/
   │   │   └── *.jpg           # images
   │   ├── cat/
   │   │   └── *.jpg
   │   └── ... (10 animal classes)
   └── test/
       ├── butterfly/
       │   └── *.jpg
       └── ... (same classes)
   ```

## Usage

```bash
python3 main.py
```

## Results
- This script trains and evaluates both the **VGG** and **AlexNet** models using the **Animals-10** dataset.
- After training, it generates visualizations including:
  - Accuracy and loss curves
  - Confusion matrices
  - A final comparison table of results between models