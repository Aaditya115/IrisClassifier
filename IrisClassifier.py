# Importing the necessary libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris  # Only import the Iris dataset now
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a dataframe to display the data
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[t] for t in y]

# Streamlit Layout
st.title('Random Forest Classifier on Iris Dataset')

st.subheader('Iris Dataset Overview')
st.write("The Iris dataset contains measurements of iris flowers and includes three species:")
st.write("- **Setosa**")
st.write("- **Versicolor**")
st.write("- **Virginica**")
st.write("Each flower has four features: sepal length, sepal width, petal length, and petal width.")
st.write(df.head())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest model
st.subheader('Train a Random Forest Classifier')
n_estimators = st.slider('Number of Trees (Estimators)', 10, 200, 100)
max_depth = st.slider('Maximum Depth of Trees', 1, 20, 10)

# Train the model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.write("This means the model was able to predict the species correctly this percentage of the time.")

# Explanation of Random Forest Concepts
st.subheader('What is Random Forest?')

st.write("""
A **Random Forest** is like a team of decision makers (called trees). Each tree looks at the data and makes a decision. The more trees (estimators) you have, the better the team is at making accurate decisions.
- **Estimators**: These are the individual decision trees in the forest. Each tree makes its own prediction, and all the trees work together to make a final prediction.
- **Max Depth**: This is how deep each tree goes. A deeper tree can make more complex decisions, but if it's too deep, it might just memorize the training data (this is called overfitting). We try to find a balance where the tree can learn the patterns without memorizing everything.
""")

# Visualizing the first Decision Tree
st.subheader('Visualizing One Decision Tree in the Random Forest')
st.write("Below is one decision tree from the Random Forest. This tree makes decisions by looking at the features (like sepal length, petal width, etc.) and deciding which species of flower it is.")

fig, ax = plt.subplots(figsize=(12, 8))
from sklearn.tree import plot_tree
plot_tree(model.estimators_[0], filled=True, feature_names=feature_names, class_names=target_names, ax=ax)
ax.set_title('Decision Tree #1 in the Random Forest')
st.pyplot(fig)

# Displaying Predictions vs Actual Labels
st.subheader('Predictions vs Actual Labels')

# Show a few predictions and actual values
predictions_df = pd.DataFrame({
    'Actual': [target_names[t] for t in y_test[:10]],  # Actual labels for first 10 test samples
    'Predicted': [target_names[t] for t in y_pred[:10]]  # Predicted labels for first 10 test samples
})

st.write(predictions_df)
st.write("This table shows the first 10 predictions made by the model compared to the actual species.")
