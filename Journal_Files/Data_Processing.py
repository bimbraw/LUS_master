import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from joblib import dump, load

dataset = pd.read_csv('Data_Xihan_11-5_components/angle_u_6.csv')
#added condition to clean up the data
#thanks to https://stackoverflow.com/questions/49546428/obtained-a-weird-strange-repeated-svr-plot-instead-on-a-single-smooth-svr-curve
#dataset = dataset.mean(axis=0)
dataset = dataset.iloc[:, 1:3]
means = dataset.groupby('x').mean()
print(means)

print('Relevant columns for dataset - ')
print(dataset)
dataset = dataset.sort_values(by=dataset.columns[1])
print('Sorted values for the dataset - ')
print(dataset)
X = dataset.iloc[:, 1].values
print('X values -')
print(X)
X = X.reshape(len(X), 1)
y = dataset.iloc[:, 0].values
dataset = dataset.mean(axis=0)
print('y values -')
print(y)
y = y.reshape(len(y), 1)

#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)
#X = X.fit_transform(X)
#y = y.fit_transform(y)

#regressor = SVR(kernel = 'rbf')
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#regressor.fit(X, y)
#y_rbf = svr_rbf.fit(X, y).predict(X)

# Predicting a new result
#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[36.5]]))))

#dump(svr_rbf, 'trained_models/angle_u_6.joblib')

#lw = 1
plt.scatter(y, X, color='pink', label='data')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('X')
plt.ylabel('y')

#plt.xlabel('Shoulder to Shoulder Width (in cm)')
#plt.ylabel('Overlay distance from center (in cm)')
plt.title('Data plotting')
plt.legend()
plt.show()