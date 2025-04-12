# 🧠 Eye Disease Classification Using Deep Learning

## 📍 Project Summary

This project presents a deep learning approach to **automatically classify retinal images** into four categories:

- **Glaucoma**
- **Cataract**
- **Normal**
- **Diabetic Retinopathy**

Using **transfer learning with the VGG19 CNN architecture**, the model achieves high accuracy and robust performance on unseen data. This solution demonstrates the power of AI in supporting early detection of eye diseases — a critical task in healthcare diagnostics.

---

## 🚀 Key Highlights

- ✅ **Model**: Transfer learning with **VGG19**
- 🖼️ **Dataset**: 3,200+ curated retinal images from Kaggle
- ⚙️ **Tech Stack**: Python, TensorFlow/Keras, NumPy, Matplotlib
- 🔄 **Techniques**: Image augmentation, Early stopping, Checkpointing
- 📈 **Outcome**: High accuracy and strong classification performance across all four disease classes

---

## 📂 Dataset

- **Source**: [Kaggle – Eye Diseases Classification Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- **Classes**: Glaucoma, Cataract, Normal, Diabetic Retinopathy
- **Preprocessing**:
  - Image resizing and normalization
  - Data augmentation: rotation, zoom, horizontal flip
  - Organized into training and validation sets via image generators

---

## 🧠 Model Architecture

- Base Model: **VGG19** (pre-trained on ImageNet)
- Added Layers:
  - Global Average Pooling
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Softmax output layer (4 classes)

---

## 🏋️ Training Strategy

- **Loss Function**: Categorical Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Callbacks**: EarlyStopping, ModelCheckpoint

---

## 📊 Results

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Confusion Matrix**: Shows balanced performance across classes

| Metric              | Score   |
|---------------------|---------|
| Training Accuracy   | 95%     |
| Validation Accuracy | 92%     |
| F1-Score (avg)      | 0.91    |

![Accuracy Plot](https://github.com/somaiaahmed/Eye-diseases-classification/assets/52898207/c1759152-ee04-417d-b61c-3b2369a85eeb)

---

## 📈 Model Evaluation

The model’s predictions were evaluated using:

- Classification Report
- Confusion Matrix
- Accuracy/Loss Curves

![Evaluation Metrics](https://github.com/somaiaahmed/Eye-diseases-classification/assets/52898207/cd10f3aa-88aa-43f4-bdef-d2f4ec1e883b)

---

## 🛠 Tools & Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

---

## 📌 Key Takeaways

- Demonstrated how **deep learning can automate medical image diagnosis**
- Achieved strong performance using **transfer learning**
- Highlighted importance of **data preprocessing and augmentation**
- Built with production-readiness and reproducibility in mind

---

## 🔗 Project Status

✅ Completed & Open for improvement  
📌 Potential future upgrades:
- Deploy model via **Streamlit or Flask** for clinical usability  
- Explore **model interpretability** (e.g., Grad-CAM heatmaps)

---

## 💼 About Me

**Stanley Ekene**  
MSc Data Science | AI Researcher | Machine Learning Developer  
🔗 [LinkedIn](https://linkedin.com/in/stanleyekene) | 🌐 [Portfolio](https://london-navigator.netlify.app) | 💻 [GitHub](https://github.com/your-github)
