# 🧠 Task 1: Dataset Preparation & Preprocessing

This repository contains Python scripts to prepare multi-class segmentation masks from the COCO 2017 dataset. The masks are suitable for training deep learning models for image segmentation tasks.

---

## 📁 Dataset

We used the **COCO 2017 validation set**, consisting of:
- `val2017/` images
- `annotations/instances_val2017.json` for segmentation labels

---

## ⚙️ Preprocessing Pipeline

The script does the following:
1. Loads images and annotations from the COCO dataset.
2. Generates multi-class segmentation masks (one class per pixel).
3. Handles key edge cases:
   - Missing or null segmentation fields
   - Zero-area annotations
   - Overlapping masks (latest overwrites earlier)
   - Invalid or unknown category IDs
4. Saves:
   - Masks in `masks/`
   - `category_mapping.json` (category ID → label index)
   - `image_mask_map.json` (image file → mask file)

---

## 📦 Output Structure

- `masks/`  
  Contains all generated multi-class segmentation masks in `.png` format.  
  Example:
  - `000000000139.png`
  - `000000000285.png`
  - `...`

- `category_mapping.json`  
  Maps COCO category IDs to continuous label indices used in masks.

- `image_mask_map.json`  
  Maps original image filenames (e.g., `000000000139.jpg`) to corresponding mask filenames.

---

## 🐧 Reproducibility on Linux (Step 3)

This project is fully reproducible on any Linux system. Run the following steps in your terminal:

   ```bash
   # 1. Clone the repository
   git clone https://github.com/ankush-raj-iisc/coco-segmentation-toolkit.git
   cd coco-segmentation-toolkit
   
   # 2. Install uv and create virtual environment
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv
   source .venv/bin/activate
   
   # 3. Install project dependencies
   uv pip install -r requirements.txt
   
   # 4. Run the notebook
   jupyter notebook python_code.ipynb
   ```


   

---
## 🛠 Environment Management with uv (Step 4)

We used `uv` for lightweight and reproducible Python dependency management.

- `pyproject.toml` defines the dependency list.
- `requirements.txt` is auto-generated using `uv pip freeze`.
- `.venv/` (local) is the `uv`-managed virtual environment.

### 🔁 To install the same environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```




---

# 🧠 Task 2: Train an Image Segmentation Model

In this task, we trained an image segmentation model using PyTorch on the preprocessed COCO dataset from Task 1. The model is designed to generate multi-class segmentation masks, generalize to unseen data, and stay within computational constraints.

---

## 🟦 Step 1: Model Setup and Training for Multi-Class Segmentation

### 📦 Custom Dataset Class: `CocoSegmentationDataset`

We created a custom PyTorch dataset to load the COCO validation images and their corresponding segmentation masks.

- Uses a JSON mapping of image-to-mask file names
- Resizes both images and masks to 256×256
- Converts images to tensors and masks to LongTensors
- Each pixel in the mask holds the class index for segmentation

---

### 🔄 Dataset Preparation and DataLoader Setup

This block handles the dataset transformation and splitting.

- **Transformations**: Resize to 256×256 and convert to tensor
- **Dataset Initialization**: Custom dataset using image and mask paths
- **Train-Validation Split**: 80% training, 20% validation
- **DataLoaders**:
  - `train_loader` with shuffling
  - `val_loader` without shuffling for consistency

---

### 🖼️ Visualize a Sample Image and its Segmentation Mask

We visualized one image from the training set to confirm:
- Left: Original RGB image
- Right: Corresponding segmentation mask (with class labels as pixel values)

This helped verify correct alignment between input images and labels.

---

## 🟦 Step 2: Demonstrate Generalization on Unseen Data

### 🔹 1. Load Pretrained DeepLabV3

We loaded a DeepLabV3 model with a ResNet-50 backbone pretrained on COCO.  
The final classifier layer was modified to output **81 channels** (80 classes + background).  
The model was moved to GPU for accelerated training.

---

### 🔹 2. Define Loss Function & Optimizer

- **Loss**: `CrossEntropyLoss` for pixel-wise multi-class classification
- **Optimizer**: `Adam` with learning rate 1e-4 for smooth fine-tuning

---

### 🔹 3. Training Loop

We wrote a training loop that:
- Enables training mode
- Loads batches of image-mask pairs
- Computes loss and performs backpropagation
- Updates model weights
- Tracks average training loss across all batches

---

### 🔹 4. Validation Accuracy Function

The evaluation function:
- Runs in inference mode (no gradients)
- Predicts the segmentation mask
- Compares pixel-wise accuracy with ground truth
- Returns overall validation accuracy %

---

### 🔹 5. Train and Evaluate for Multiple Epochs

The model was trained for **5 epochs** on the full dataset.  
For each epoch, we printed:
- Average training loss
- Pixel-wise validation accuracy

This confirmed that the model learns progressively and generalizes well.

---

### 🔹 6. Visualize Predictions vs Ground Truth

We visualized:
- The **input image**
- The **ground truth mask**
- The **predicted mask**

This helped qualitatively verify that the model segments object boundaries and categories correctly.

---

## ⏱️ Model Runtime

- Training was performed on a GPU-based notebook environment.
- **Total training time: ~4.5 hours**, well under the 6-hour constraint.
- No use of `ultralytics` or external pretrained segmentation APIs. The entire model pipeline is built using **pure PyTorch and torchvision**.

---





