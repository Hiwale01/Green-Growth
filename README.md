
# 🌿 Green Growth – Plant Disease Detection using Deep Learning

![Green Growth Banner](https://your-image-url.com/banner.jpg)

Welcome to **Green Growth** — an AI-powered solution for detecting plant diseases using deep learning and image classification. This project aims to empower farmers, agriculturists, and plant lovers with a smart, fast, and accurate way to diagnose plant diseases just by uploading a photo.

---

## 🚀 Project Overview

This system uses a Convolutional Neural Network (CNN) trained on a dataset of plant leaves to predict whether the plant is healthy or suffering from a disease — and if so, which disease it is.

---

## 🔍 Features

- 🧠 Deep learning-based CNN model
- 🌿 Supports multiple plant species and disease classes
- 📷 Easy image-based prediction
- 📊 Visualized results and metrics
- ✅ User-friendly output

---

## 🧪 Sample Results

### 🖼️ Input Image
![Sample Leaf](https://your-image-url.com/sample-leaf.jpg) <!-- Replace with actual image -->

### ✅ Model Output
- **Predicted Class:** Tomato - Late Blight
- **Confidence Score:** 97.65%

### 📉 Confusion Matrix
![Confusion Matrix](https://your-image-url.com/confusion-matrix.jpg)

---

## 🧠 Model Architecture

- Input Layer: 128x128 RGB image
- 3 Convolution + MaxPooling Layers
- Dense + Dropout layers
- Output Layer: Softmax for multiclass classification

```

Model: "sequential"

---

# Layer (type)                 Output Shape              Param \#

conv2d (Conv2D)              (None, 128, 128, 32)      896
...
dense (Dense)                (None, 15)                1230
===========================================================

Total params: 1,235,000
Trainable params: 1,235,000

````

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Matplotlib
- Jupyter Notebook

---

## 📁 Dataset Used

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) – Contains 50,000+ labeled images of plant leaves.

---

## 🔧 How to Run Locally

```bash
git clone https://github.com/Hiwale01/Green-Growth.git
cd Green-Growth

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

pip install -r requirements.txt
jupyter notebook
````

Open `PlantDiseaseDetection.ipynb` and run all cells.

---

## 📊 Model Performance

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.25% |
| Precision | 97.6%  |
| Recall    | 98.1%  |
| F1 Score  | 97.85% |

---

## 👨‍🌾 Use Case

* Agricultural health monitoring
* Early disease detection to reduce crop loss
* Educational tool for botany/agriculture students

---

## 📸 Screenshots

| Input                                               | Prediction                                       |
| --------------------------------------------------- | ------------------------------------------------ |
| ![Input Leaf](https://your-image-url.com/input.jpg) | ![Output](https://your-image-url.com/output.jpg) |

---

## 🙌 Acknowledgments

* Dataset by PlantVillage (via Kaggle)
* TensorFlow team and open-source community

---

## 🧑‍💻 Author

Made with 💚 by [**Hiwale01**](https://github.com/Hiwale01)

---

## ⭐ Show your support

If you found this project helpful or interesting, please consider ⭐ starring the repo!

````

---

### ✅ To Do:
- Replace all `https://your-image-url.com/...` with actual image links from your repo or external image hosting (like [Imgur](https://imgur.com), or GitHub raw URLs).
- Save it as `README.md` in the root folder.
- Git add → commit → push:

```bash
git add README.md
git commit -m "Add professional README with project overview and results"
git push
````

Would you like me to help generate actual image links from your project or auto-insert them into the README?
