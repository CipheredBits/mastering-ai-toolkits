# 🧠 Mastering AI Toolkits – Final Project

This project showcases the practical use of three major AI toolkits — **Scikit-learn**, **TensorFlow**, and **spaCy** — through real-world machine learning and NLP tasks. The goal is to demonstrate core AI capabilities in structured data classification, deep learning with images, and natural language processing.

---

## 🔹 1. Classifying Iris Flowers with Decision Trees  
**Toolkit:** Scikit-learn  
**Dataset:** Iris Species (from sklearn)

This task involved training a Decision Tree Classifier to predict the species of iris flowers based on features like petal length and sepal width.

### 📈 Key Highlights:
- Dataset preprocessed and encoded for modeling
- Decision Tree trained and tested using `train_test_split`
- Evaluation with **accuracy, precision, recall**, and full classification report

### ✅ Result:
- Achieved **100% accuracy** on the test set
- Perfect scores on all evaluation metrics
- Verified the strong separability of the Iris dataset

> **Insight:** While ideal performance is encouraging, such results highlight the dataset’s simplicity. Real-world deployment would require additional validation steps like cross-validation to avoid overfitting.

---

## 🔹 2. Digit Recognition with a Convolutional Neural Network  
**Toolkit:** TensorFlow/Keras  
**Dataset:** MNIST (handwritten digits)

A deep learning model was designed using a Convolutional Neural Network (CNN) to recognize digits from 28x28 grayscale images.

### ⚙️ Architecture:
- Convolutional + MaxPooling layers
- Flatten and fully connected (Dense) layers
- Softmax output for 10-digit classification

### ✅ Result:
- Achieved **~99% accuracy** on the test set
- Stable training curves with no sign of overfitting
- Confident classification of handwritten digits

> **Insight:** CNNs are highly effective for image classification when trained with sufficient data. Even basic architectures can perform exceptionally on structured datasets like MNIST.

---

## 🔹 3. Product Review Analysis with NER and Sentiment Logic  
**Toolkit:** spaCy  
**Dataset:** Simulated Amazon reviews

This task involved analyzing customer reviews using Named Entity Recognition and a custom rule-based sentiment analyzer.

### 📊 Features:
- spaCy NER extracts products, brands, and other entities
- Rule-based sentiment logic uses positive/negative word lists
- Handles intensifiers and negations for better context understanding
- Visual summaries of entity frequency and sentiment distribution

### ✅ Result:
- Accurate sentiment classification aligned with labeled data
- Identified product and brand mentions effectively
- Insightful visualizations (pie charts, bar plots)

> **Insight:** Rule-based systems, though limited in depth, are transparent and fast. Combined with spaCy’s entity recognition, they offer a solid foundation for quick text analysis in production-like settings.

---

## 🧾 Summary

Each toolkit was applied to a different branch of AI:
- **Scikit-learn**: Quick and interpretable classical models
- **TensorFlow**: Robust deep learning for image recognition
- **spaCy**: Efficient text processing and custom NLP logic

The project demonstrates how different tools excel in specific domains, and how combining them leads to a broad, practical understanding of modern AI workflows.
