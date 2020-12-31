import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import addcopyfighandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm

folder = 'angle_v_7'
xlabel_u = 'U values'
xlabel_v = 'V values'

u_or_v = 'v'
kernel = 'poly'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder + '.csv')
print(dataset)
print(dataset.shape)
X = dataset.iloc[:, 1].values
X = X.reshape(len(X), 1)
y = dataset.iloc[:, 2].values

#test training split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#added condition to clean up the data
#thanks to https://stackoverflow.com/questions/49546428
dataset = dataset.iloc[:, 1:3]

#https://www.geeksforgeeks.org/python-ways-to-remove-duplicates-from-list/
y_compressed = []
for i in y:
    if i not in y_compressed:
        y_compressed.append(i)

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

val_y = []
for i in means.index:
    val_y.append(i)

print('Here is the val_y list -')
print(val_y)

val_x = []
for i in means.angle:
    val_x.append(i)

print('Here is the val_x list -')
print(val_x)

X_train, X_test, y_train, y_test = train_test_split(val_x, val_y, test_size=0.2, random_state=0)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
print('Training set - X')
print(X_train.shape)
print(X_train)
print('Test set - X')
print(X_test.shape)
print(X_test)
print('Training set - y')
print(y_train.shape)
print(y_train)
print('Test set - y')
print(y_test.shape)
print(y_test)


X_train = X_train.astype('double')
y_train = y_train.astype('double')
X_test = X_test.astype('double')
y_test = y_test.astype('double')


print('Started Model training')
clf = svm.SVR(kernel=kernel).fit(X_train, y_train)
print(clf)
print('Model trained')

#make predictions
pred = clf.predict(X_test)

print('Here is the mean squared error -')
print(mean_squared_error(pred, y_test))

fig = plt.figure()
#print(X.reshape(1, -1))
#print(y)
plt.scatter(X.reshape(1, -1), y, color='yellow', label='original data')
plt.scatter(val_x, val_y, color='pink', label='mean data')
plt.plot(means, y_compressed, color='red', label='connected mean data')
if u_or_v == 'u':
    plt.ylabel(xlabel_u)
if u_or_v == 'v':
    plt.ylabel(xlabel_v)
plt.xlabel('Angle')
#plt.ylim(0, 255)
#plt.xlim(-50, 50)
plt.legend()
plt.title('Output plot for ' + folder)
plt.show()

fig = plt.figure()
#print(X_test.shape)
#print(y_test)
pred = np.reshape(pred, (-1, 1))
#print(pred.shape)
#print(pred)

test_data = np.concatenate((np.array(X_test), np.array(y_test)), axis=1)
test_data = pd.DataFrame(test_data)
sorted_test = test_data.sort_values(by=test_data.columns[0])
#print('Here is the sorted test data -')
#print(sorted_test)
pred_data = np.concatenate((np.array(X_test), np.array(pred)), axis=1)
pred_data = pd.DataFrame(pred_data)
sorted_pred = pred_data.sort_values(by=pred_data.columns[0])
#print('Here is the sorted pred data -')
#print(sorted_pred)
#print(test_data[:, 1])
#print(test_data)
#print(sorted_pred[0])
plt.scatter(sorted_test[0], sorted_test[1], color='black', label='test data')
plt.plot(sorted_pred[0], sorted_pred[1], color='blue', label='prediction curve')
if u_or_v == 'u':
    plt.ylabel(xlabel_u)
if u_or_v == 'v':
    plt.ylabel(xlabel_v)
plt.xlabel('Angle')
#plt.ylim(0, 255)
#plt.xlim(-50, 50)
plt.legend()
plt.title('Test data/prediction data ' + folder)
plt.show()