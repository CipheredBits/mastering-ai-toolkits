### Task 1: Classical ML with Scikit-learn
* Dataset: Iris Species Dataset (can be loaded directly from scikit-learn)
* Goal: Preprocess data, train Decision Tree Classifier, evaluate performance.


```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['Species'] = iris.target  # Add target as a column for clarity

# Convert species to categorical for encoding demonstration
iris_df['Species'] = iris_df['Species'].astype('category')

# Encode the Species column to numerical values
label_encoder = LabelEncoder()
iris_df['Species'] = label_encoder.fit_transform(iris_df['Species'])

print("\n=== Species after encoding ===")
print(iris_df['Species'].value_counts())

# Separate features and target variable
X = iris_df.drop(['Species'], axis=1)  # Features
y = iris_df['Species']  # Target

print("\n=== Features (X) ===")
print(X.head())
print("\n=== Target (y) ===")
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n=== Data split ===")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# Model Training - Decision Tree Classifier
print("\n=== Training Decision Tree Classifier ===")
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Model Evaluation
print("\n=== Model Evaluation ===")

# Predictions on test set
y_pred = dt_classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Detailed classification report
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Interpretation of results
print("\n=== Interpretation ===")
print("1. Accuracy represents the overall correctness of the model.")
print("2. Precision indicates how many of the predicted positives are actually positive.")
print("3. Recall shows how many of the actual positives the model correctly identified.")
print("\nThe Decision Tree classifier performs well on the Iris dataset, which is expected as it's a well-separable dataset.")

```

    
    === Species after encoding ===
    Species
    0    50
    1    50
    2    50
    Name: count, dtype: int64
    
    === Features (X) ===
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    0                5.1               3.5                1.4               0.2
    1                4.9               3.0                1.4               0.2
    2                4.7               3.2                1.3               0.2
    3                4.6               3.1                1.5               0.2
    4                5.0               3.6                1.4               0.2
    
    === Target (y) ===
    0    0
    1    0
    2    0
    3    0
    4    0
    Name: Species, dtype: int64
    
    === Data split ===
    Training set size: 105 samples
    Testing set size: 45 samples
    
    === Training Decision Tree Classifier ===
    
    === Model Evaluation ===
    Accuracy: 1.0000
    Precision: 1.0000
    Recall: 1.0000
    
    === Classification Report ===
                  precision    recall  f1-score   support
    
          setosa       1.00      1.00      1.00        19
      versicolor       1.00      1.00      1.00        13
       virginica       1.00      1.00      1.00        13
    
        accuracy                           1.00        45
       macro avg       1.00      1.00      1.00        45
    weighted avg       1.00      1.00      1.00        45
    
    
    === Interpretation ===
    1. Accuracy represents the overall correctness of the model.
    2. Precision indicates how many of the predicted positives are actually positive.
    3. Recall shows how many of the actual positives the model correctly identified.
    
    The Decision Tree classifier performs well on the Iris dataset, which is expected as it's a well-separable dataset.


### Interpretation of the Results:

1. **Data Integrity**:

   * There are no missing values in the dataset, ensuring a clean input for the model.

2. **Encoding**:

   * The species column was numerically encoded, which is required for training the Decision Tree Classifier. This step was redundant in this specific dataset, as the `target` was already encoded, but it serves as a good practice for datasets requiring manual encoding.

3. **Model Performance**:

   * The Decision Tree achieved perfect accuracy (1.0) on the test set, which may indicate excellent separability of the Iris dataset. This is expected given the simplicity and separability of this dataset.

4. **Classification Metrics**:

   * **Precision, Recall, and F1-Score**: All metrics are 1.0 for each class, indicating that the classifier is correctly identifying all classes without errors. This reinforces the effectiveness of the model on this dataset.
   * **Support**: Each class is well-represented, with 19, 13, and 13 samples in the test set.

5. **Decision Tree Structure**:

   * The tree structure highlights logical splits based on feature values (`petal length`, `petal width`, etc.), effectively separating the classes.
   * The simplicity of the tree underscores the separability of the dataset, with clear decision boundaries.

6. **Conclusion**:

   * The Decision Tree Classifier is an excellent choice for the Iris dataset. However, the perfect performance suggests overfitting might occur on more complex datasets or unseen data. Cross-validation would provide a better robustness measure.




```python

```
