# example of comparing number of cores used during training to execution speed
from time import time
import mlxtend
from mlxtend.evaluate import accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=12000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
results = list()
# compare timing for number of cores
n_cores = [1, 2, 3, 4, 5, 6]
for n in n_cores:
	# capture current time
	start = time()
	# define the model
	model = RandomForestClassifier(n_estimators=600, n_jobs=n)
	# fit the model
	model.fit(X_train, y_train)
	# capture current time
	end = time()
	# store execution time
	result = end - start
	# make predictions
	yhat = model.predict(X_test)
	# evaluate predictions
	acc = accuracy_score(y_test, yhat)
	print('Accuracy: %.3f' % acc)
	print('>cores=%d: %.3f seconds' % (n, result))
	results.append(result)
pyplot.plot(n_cores, results)
pyplot.show()