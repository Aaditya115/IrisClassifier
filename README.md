# Iris Dataset - Random Forest Classifier with Streamlit 🌸🌿

This project demonstrates how to use a **Random Forest Classifier** to predict the species of iris flowers based on their measurements from the **Iris dataset**. Built with **Streamlit**, this interactive web app allows users to experiment with the model by adjusting hyperparameters and visualize how the Random Forest Classifier works in real time.

---

## 🛠️ Key Features

- **Interactive Model Training**: Adjust the number of trees (estimators) and the maximum depth of each decision tree to observe how these changes affect the model's performance.
- **Model Accuracy**: View the accuracy of the trained Random Forest model based on predictions made on a test set of the Iris dataset.
- **Decision Tree Visualization**: Visualize one decision tree from the Random Forest to understand how the model makes its predictions.
- **Predictions vs Actual Labels**: See a table comparing the model's predictions with the actual species labels for the first 10 test samples.

---

## 📊 How the App Works

### 1. **Dataset Overview**  
The app starts by displaying an overview of the **Iris dataset**, which contains measurements of three species of iris flowers:

- **Setosa**
- **Versicolor**
- **Virginica**

Each flower has four features: sepal length, sepal width, petal length, and petal width.

### 2. **Train a Random Forest Classifier**  
Users can adjust the **number of trees (estimators)** and the **maximum depth** of each decision tree. The app will train a Random Forest model using these hyperparameters and compute the **accuracy** of predictions on a test set.

### 3. **Visualizing One Decision Tree**  
The app visualizes one of the individual decision trees from the Random Forest to give a clear idea of how decisions are made based on the features.

### 4. **Predictions vs Actual Labels**  
The app shows a table comparing the predicted flower species with the actual species for the first 10 samples in the test set.

---
