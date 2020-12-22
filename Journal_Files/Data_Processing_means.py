import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import addcopyfighandler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn import svm

folder = 'angle_v_1'
xlabel_u = 'U values'
xlabel_v = 'V values'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder + '.csv')
print(dataset)
print(dataset.shape)
X = dataset.iloc[:, 2].values
X = X.reshape(len(X), 1)
y = dataset.iloc[:, 1].values

#test training split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#added condition to clean up the data
#thanks to https://stackoverflow.com/questions/49546428
dataset = dataset.iloc[:, 1:3]

#https://www.geeksforgeeks.org/python-ways-to-remove-duplicates-from-list/
X_compressed = []
for i in X:
    if i not in X_compressed:
        X_compressed.append(i)

#added mean condition thanks to
#https://stackoverflow.com/questions/64879466
means = dataset.groupby('val').mean()
print(means.shape)

print('Here is the sorted dataframe -')
print(means)

#this attribute is a list
print('axes attribute of the Dataframe -')
print(means.axes)

#this attribute of the dataframe data structure
print('index attribute of the Dataframe -')
print(means.index)

#this is class 'pandas.core.indexes.numeric.Int64Index'
print('Printing the first element -')
print(means.index[0])

val_x = []
for i in means.index:
    val_x.append(i)

print('Here is the val_x list -')
print(val_x)

val_y = []
for i in means.angle:
    val_y.append(i)

print('Here is the val_y list -')
print(val_y)

#holdout_score = -1 * metrics.mean_squared_error(val_y, val_x)
#print(holdout_score)

X_train, X_test, y_train, y_test = train_test_split(val_x, val_y, test_size=0.2, random_state=0)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
print('Training set - X')
print(X_train.shape)
print(type(X_train))
print(X_train)
print('Test set - X')
print(X_test.shape)
print(type(X_test))
print(X_test)
print('Training set - y')
print(y_train.shape)
print(type(y_train))
print(y_train)
print('Test set - y')
print(y_test.shape)
print(type(y_test))
print(y_test)

#print(X_train.reshape(1, -1))
#print(y_train.reshape(1, -1))
#print(X_test.reshape(1, -1))
#print(y_test.reshape(1, -1))

X_train = X_train.astype('double')
y_train = y_train.astype('double')
X_test = X_test.astype('double')
y_test = y_test.astype('double')

print('Started Model training')
clf = svm.SVR(kernel='linear').fit(X_train, y_train)
print('Model trained')

#The coefficient of determination (R2 score)
print('Here is the score -')
print(clf.score(X_test, y_test))

fig = plt.figure()
plt.scatter(X, y, color='pink', label='data')
plt.plot(X_compressed, means)
plt.xlabel(xlabel_v)
plt.ylabel('Angle')
plt.xlim(0, 255)
plt.ylim(-50, 50)
plt.title('Output plot for ' + folder)
plt.savefig('C:/Users/bimbr/OneDrive/Desktop/Research/Lung_Ultrasound/Updated_plots/' + folder + '.png')
plt.show()