# Animal Classification - Image Preprocessing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset:**
   # The dataset is already included in the .zip file in the correct directory
    
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

   # If you chose to download it from scratch
   - Download the dataset from kaggle, https://www.kaggle.com/datasets/viratkothari/animal10
   - Put the dataset `Animals-10/` folder in the same directory as the main.py file and the split_data file
   - Run the "split_data.py" python script to split the data into test & train folders
   ```bash
   python3 split_data.py
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