# 🧠 COCO Segmentation Dataset Preprocessing

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




