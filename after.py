import importlib
import regression
import datasets
import numpy as np

import matplotlib.pyplot as plt

importlib.reload(regression)
X,Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)

model = regression.RidgeRegression(alpha=0)
#model = regression.RidgeRegression()
model.fit(ex_X,Y)

print(model.theta)
print(model.predict(ex_X))
print(model.score(ex_X,Y))


samples = np.arange(0,4,0.1)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = datasets.polynomial3_features(x_samples)


plt.scatter(X[:,1],Y)
plt.plot(samples,model.predict(ex_x_samples))

model = regression.RidgeRegression(alpha=0.1)
model.fit(ex_X,Y)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = datasets.polynomial3_features(x_samples)
plt.plot(samples,model.predict(ex_x_samples))

model = regression.RidgeRegression(alpha=0.5)
model.fit(ex_X,Y)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = datasets.polynomial3_features(x_samples)
plt.plot(samples,model.predict(ex_x_samples))

model = regression.RidgeRegression(alpha=1)
model.fit(ex_X,Y)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = datasets.polynomial3_features(x_samples)
plt.plot(samples,model.predict(ex_x_samples))

model = regression.RidgeRegression(alpha=10)
model.fit(ex_X,Y)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = datasets.polynomial3_features(x_samples)
plt.plot(samples,model.predict(ex_x_samples))
plt.show()