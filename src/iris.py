from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# scale the data
X = StandardScaler().fit_transform(X)

# Instantiate the Estimator
model = LogisticRegression()

# fit the model to the data
print(model.fit(X, y))

# Make a prediction
predictions = model.predict(X[:130])
print(predictions)
