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






