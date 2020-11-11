import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from joblib import dump, load

dataset = pd.read_csv('Data_Xihan_11-5_components/angle_v_5.csv')
#added condition to clean up the data
#thanks to https://stackoverflow.com/questions/49546428/obtained-a-weird-strange-repeated-svr-plot-instead-on-a-single-smooth-svr-curve
dataset = dataset.sort_values(by=dataset.columns[1])
X = dataset.iloc[:, -2].values
print('X values -')
X = X.reshape(len(X), 1)
y = dataset.iloc[:, -1].values
print('y values -')
print(y)
y = y.reshape(len(y), 1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor = SVR(kernel = 'rbf')
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
regressor.fit(X, y)
y_rbf = svr_rbf.fit(X, y).predict(X)

# Predicting a new result
#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[36.5]]))))

dump(svr_rbf, 'trained_models/angle_v_5.joblib')

lw = 1
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('Angle')
plt.ylabel('Ratio')

plt.xlabel('Shoulder to Shoulder Width (in cm)')
plt.ylabel('Overlay distance from center (in cm)')
plt.title('Fittings and data plotting')
plt.legend()
plt.show()