import importlib
import regression
import datasets

importlib.reload(regression)
X,Y = datasets.load_linear_example1()

model = regression.LinearRegression()
print(model.fit(X,Y))
print(model.score(X,Y))