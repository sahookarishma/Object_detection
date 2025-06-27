# YOLO Object Detection: Cats and Dogs

This project implements a YOLO-style object detector for detecting cats and dogs in images using PyTorch. The model is trained from scratch on a custom dataset with Pascal VOC-style XML annotations.

## Features
- Custom YOLO architecture in PyTorch
- Supports multiple anchors and classes
- Training and evaluation scripts
- Calculates mAP (mean Average Precision)

---

## Dataset Structure

```
object-detection/
├── dataset/
│   ├── images/         # Training images (e.g., Cats_Test1.png)
│   └── annotation/     # Pascal VOC XML annotation files (e.g., Cats_Test1.xml)
├── codes.ipynb         # Main Jupyter notebook
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

- Each image in `images/` should have a corresponding XML file in `annotation/`.
- Classes: `cat` (0), `dog` (1)

---

## Setup

1. **Clone the repository**
2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
   - Place your images in `dataset/images/`
   - Place your Pascal VOC XML files in `dataset/annotation/`

---

## Training

Open `codes.ipynb` in Jupyter Notebook or JupyterLab and run all cells. The notebook will:
- Define the YOLO model and loss
- Load and preprocess the dataset
- Train the model for a specified number of epochs
- Save the trained model as `yoloObjectdetection_cat_dog_model.pth`

**Example training output:**
```
Epoch 1 Loss: 484.0259
Epoch 2 Loss: 1119.0332
...
Epoch 10 Loss: 27.9706
Model saved successfully!
```

---

## Evaluation

The notebook includes code to evaluate the trained model and compute mAP:
- Decodes predictions and targets
- Applies non-max suppression
- Calculates AP for each class and overall mAP

**Example evaluation output:**
```
Evaluating model on training data...
AP for cat: 0.1234
AP for dog: 0.2345
mAP: 0.1789
```

---

## Inference

You can add your own inference code to visualize predictions on new images. Example snippet:
```python
# Load model
model = YOLO(grid_size=7, num_classes=2, num_anchors=3)
model.load_state_dict(torch.load('yoloObjectdetection_cat_dog_model.pth'))
model.eval()

# Prepare image and run prediction...
```

---

## Dependencies
- Python 3.7+
- torch
- torchvision
- numpy
- opencv-python
- tqdm
- matplotlib

Install all dependencies with:
```
pip install -r requirements.txt
```

---

## Notes
- The model and code are for educational purposes and may require further tuning for production use.
- For best results, use a larger and more diverse dataset.
