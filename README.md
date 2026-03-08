# Enhancing-Signature-Forgery-Detection-System-using-CNN-SVM

This project focuses on developing an intelligent system to detect forged signatures using advanced machine learning techniques. Signature verification is widely used in banking, legal documentation, and identity authentication. Manual verification can be time-consuming and prone to human error, so this project aims to automate the process using deep learning and machine learning models.

The system uses a combination of **Convolutional Neural Networks (CNN)** for feature extraction and **Support Vector Machine (SVM)** for classification. CNN automatically learns important visual patterns from signature images, while SVM helps classify whether a signature is genuine or forged.

The goal of this project is to improve the **accuracy, reliability, and efficiency** of signature forgery detection compared to traditional verification methods.

---

## 📌 Project Objectives

* Detect forged signatures using machine learning techniques
* Extract meaningful features from signature images using CNN
* Classify signatures as **genuine** or **forged** using SVM
* Improve verification accuracy and reduce manual effort
* Build a scalable system that can be applied to real-world authentication systems

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

---

## 📂 Project Structure

```
Enhancing-Signature-Forgery-Detection-System-using-CNN-SVM
│
├── data
│   ├── train
│   │   ├── genuine
│   │   └── forged
│   │
│   └── validation
│       ├── genuine
│       └── forged
│
├── cnn_model
│   └── train_cnn.py
│
├── svm_model
│   └── svm_classifier.py
│
├── feature_extraction
│   └── extract_features.py
│
├── results
│   └── model_accuracy.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How the System Works

1. **Data Collection**
   Signature images are collected and divided into training and validation datasets.

2. **Preprocessing**
   Images are resized, converted to grayscale, and normalized for better model performance.

3. **Feature Extraction (CNN)**
   A Convolutional Neural Network extracts deep features from the signature images.

4. **Classification (SVM)**
   Extracted features are fed into an SVM classifier to determine whether the signature is genuine or forged.

5. **Model Evaluation**
   The model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score

---

## 🚀 Applications

* Banking signature verification
* Legal document authentication
* Identity verification systems
* Fraud detection systems

---

## 📈 Future Improvements

* Use larger and more diverse datasets
* Implement real-time signature verification
* Improve CNN architecture for better feature extraction
* Deploy the system as a web or mobile application

---

## 👨‍💻 Author

Sunil K

---

⭐ If you find this project useful, consider giving it a **star** on GitHub.
