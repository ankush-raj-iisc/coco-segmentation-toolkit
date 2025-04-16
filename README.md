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

The model was trained for **5 epochs** on the 1500-image subset.  
We recorded training loss and pixel-wise validation accuracy per epoch.

#### 📊 Training Results:

| Epoch | Training Loss | Validation Accuracy |
|-------|----------------|---------------------|
| 1     | 2.0013         | 77.31%              |
| 2     | 1.1461         | 77.59%              |
| 3     | 1.0024         | 78.71%              |
| 4     | 0.9008         | 80.39%              |
| 5     | 0.8183         | 79.90%              |

These results show a consistent decrease in loss and improvement in accuracy, indicating good learning and generalization.

---

### 🔹 6. Visualize Predictions vs Ground Truth

We visualized:
- The **input image**
- The **ground truth mask**
- The **predicted mask**

Below is a sample output from the model:

![image](https://github.com/user-attachments/assets/a39f093a-e9c7-40f6-8c9d-343c7daace60)

This helped qualitatively verify that the model segments object boundaries and categories correctly.

---

## ⏱️ Model Runtime

- Training was performed on a GPU-based notebook environment.  
- **Total training time: ~4.5 hours**, well under the 6-hour constraint.  
- No use of `ultralytics` or external pretrained segmentation APIs.  
- The entire model pipeline is built using **pure PyTorch and torchvision**.

---




