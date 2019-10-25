import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.patches import Ellipse
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
# Compare Algorithms
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def models(X,Y):
	# prepare configuration for cross validation test harness
	seed = 1
	# prepare models
	models = []
	# models.append(('LR', LogisticRegression()))
	# models.append(('LDA', LinearDiscriminantAnalysis()))
	# models.append(('KNN', KNeighborsClassifier()))
	# models.append(('CART', DecisionTreeClassifier())) ## worked best for me  for CV testing
	models.append(('NB', GaussianNB()))
	# models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	scoring = 'accuracy'
	print('training have started')
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		model.fit(X,Y)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

	return model

def save_model(model):
	filename = 'svm_model.sav'
	joblib.dump(model, filename)

def load_model(filename):
	# load the model from disk
	loaded_model = joblib.load(filename)
	result = loaded_model.score(X_test, y_test)
	print(result)
	return loaded_model


if __name__ == '__main__':


	# water = np.load('Data/water_fuse.npy')
	# grass = np.load('Data/grass_fuse.npy')
	#
	# grass = grass.transpose()
	# water= water.transpose()
	#
	# y_g = np.zeros(len(grass))
	# l = len(y_g)
	# y_g.reshape((l,1))
	#
	# y_w = np.ones(len(water))
	# y_w.reshape((len(y_w),1))
	#
	# x = np.vstack((water,grass))
	#
	# y = np.hstack((y_w,y_g))
	# y = y.reshape((len(y),1))
	#
	# data = np.hstack((x,y))
	# np.random.shuffle(data)
	data = np.load('Data/run2/DATASET.npy')
	print(np.shape(data))
	data = np.nan_to_num(data)

	from sklearn.preprocessing import StandardScaler
	X = StandardScaler().fit_transform(data[:,:-1])

	# X = data[:,:-1]
	Y = data[:,-1]

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
	# svm(X_train, X_test, y_train, y_test)

	model = models(X_train, y_train)
	result = model.score(X_test, y_test)
	print(result)
	save_model(model)


