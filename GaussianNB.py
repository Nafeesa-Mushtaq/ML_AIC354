from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# 1. Load dataset
iris = load_iris()

# 2. Separate features and target
X = iris.data
y = iris.target

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# 4. Create the model
model = GaussianNB()

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict test results
y_pred = model.predict(X_test)

# 7. Calculate accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100, "%")

# 8. Predict a new flower
print("Prediction:", model.predict([ [5.1, 3.5, 1.4, 0.2] ]))