import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.svm import SVR

dataset = pd.read_csv('Data_Xihan_11-5_components/angle_u_1.csv')
X = dataset.iloc[:, -2].values
print('X values -')
print(type(X))
X = X.reshape(len(X), 1)
#X = normalize(X)
#print(X)
y = dataset.iloc[:, -1].values
print('y values -')
print(y)
y = y.reshape(len(y), 1)
#print('Updated y values -')
#print(y)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

plt.scatter(X, y, color='darkorange', label='data')
plt.show()

regressor = SVR(kernel = 'rbf')
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
regressor.fit(X, y)
y_rbf = svr_rbf.fit(X, y).predict(X)

# Predicting a new result
#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[36.5]]))))

#lw = 1
#plt.scatter(X, y, color='darkorange', label='data')
#plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
#plt.xlabel('Angle')
#plt.ylabel('Ratio')

#plt.xlabel('Shoulder to Shoulder Width (in cm)')
#plt.ylabel('Overlay distance from center (in cm)')
#plt.title('Fittings and data plotting')
#plt.legend()
#plt.show()