import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm

folder = 'angle_u_1.csv'

u_or_v = 'u'

xlabel_u = 'U value'
xlabel_v = 'V value'
kernel = 'rbf'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder)
dataset = dataset.iloc[:, 1:3]

print('Relevant columns for dataset - ')
print(dataset)
dataset = dataset.sort_values(by=dataset.columns[1])
print('Sorted values for the dataset - ')
print(dataset)
X = dataset.iloc[:, 0].values
print('X values -')
print(X)
X = X.reshape(len(X), 1)
y = dataset.iloc[:, 1].values
print('y values -')
print(y)
y = y.reshape(len(y), 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
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

#print('Here is the mean squared error -')
#print(mean_squared_error(pred, y_test))

fig = plt.figure()
#print(X.reshape(1, -1))
#print(y)
plt.scatter(X.reshape(1, -1), y.ravel(), color='blue', label='original data')
#plt.scatter(val_x, val_y, color='pink', label='mean data')
#plt.plot(means, y_compressed, color='red', label='connected mean data')
if u_or_v == 'u':
    plt.ylabel(xlabel_u)
if u_or_v == 'v':
    plt.ylabel(xlabel_v)
plt.xlabel('Angle')
#plt.ylim(0, 255)
#plt.xlim(-50, 50)
plt.legend()
plt.title('Output plot for ' + folder)
#plt.show()

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
#plt.show()

pred = pred.squeeze()
y_test = y_test.squeeze()
print('Here is the predicted values -')
print(pred.round(1))
print('Here is the y test values -')
print(y_test.round(1))
print('Here is the difference of both values -')
difference = y_test.squeeze() - pred.squeeze()
print(difference.round(1))
print('Here is the square of the difference -')
#this is to remove scientific notation
np.set_printoptions(suppress=True)
difference_sq = (y_test.squeeze() - pred) ** 2
print(difference_sq.round(1))
print('The sum of squares of differences is -')
print(sum(difference_sq).round(1))

mse = sum(difference_sq / len(pred)).round(1)
print('Here is the mean squared error -')
print(mse)