import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import scipy.interpolate

folder = 'angle_v_1'
xlabel_u = 'U values'
xlabel_v = 'V values'

u_or_v = 'v'
#kernel = 'poly'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder + '.csv')
#print(dataset)
#print(dataset.shape)
x = dataset.iloc[:, 1].values
#x = x.reshape(len(x), 1)
y = dataset.iloc[:, 2].values
print('Here is x (angles) - ')
print(x)
print('Here is y (coordinate value) - ')
print(y)
#print(round(max(x)))
#print(max(y))

plt.figure(0)
plt.scatter(x, y)
plt.xlim(round(min(x)), round(max(x)))
plt.ylim(round(min(y)), round(max(y)))
plt.title('Input data')
#plt.show()

val_up = 0
print(len(y))

for i in range(len(y)):
    val = y[i]
    val_up = val_up + val

sorted_pairs = sorted((i, j) for i, j in zip(x, y))
true_num = round(min(x))
means = [0] * (int(round(max(x)) - round(min(x))))

index = round(min(x))
index_array = [0] * (int(round(max(x)) - round(min(x))))

for r in range(int(round(min(x))), int(round(max(x)))):
    index += 2
    true_num += 2
    true_val = 0
    summed = 0
    for q in range(len(y)):
        if sorted_pairs[q][0] < true_num and sorted_pairs[q][0] > (true_num - 2):
            summed += sorted_pairs[q][1]
            true_val += 1
        else:
            continue
    if true_val == 0:
        means[r] = 0
    else:
        means[r] = summed / true_val
    index_array[r] = index

print(means)
print(index_array)

plt.figure(1)
plt.scatter(index_array, means)
plt.xlim(round(min(x)), round(max(x)))
plt.ylim(round(min(y)), round(max(y)))
plt.title('Means data')
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(index_array, means, test_size=0.2, random_state=0)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
print('Training set - X')
print(X_train.shape)
print(X_train.reshape(1, -1))
print('Test set - X')
print(X_test.shape)
print(X_test.reshape(1, -1))
print('Training set - y')
print(y_train.shape)
print(y_train.reshape(1, -1))
print('Test set - y')
print(y_test.shape)
print(y_test.reshape(1, -1))

plt.figure(2)
plt.scatter(X_train, y_train, color='red', label='training data')
plt.scatter(X_test, y_test, color='blue', label='testing data')
plt.title('Testing and Training data')
plt.xlim(round(min(x)), round(max(x)))
plt.ylim(round(min(y)), round(max(y)))
plt.legend()
#plt.show()

# squeeze to make it 1D
y_interp = scipy.interpolate.interp1d(X_train.squeeze(), y_train.squeeze(), kind='cubic', fill_value="extrapolate")

pred = y_interp(X_test.squeeze())
print(pred)

plt.figure(3)
plt.scatter(X_test, pred, color='red', label='predicted data')
plt.scatter(X_test, y_test, color='blue', label='testing data')
plt.title('Predicted data with Testing data')
plt.xlim(round(min(x)), round(max(x)))
plt.ylim(round(min(y)), round(max(y)))
plt.legend()
#plt.show()

print('Here is the predicted values -')
print(pred.round(1))
print('Here is the y test values -')
print(y_test.squeeze().round(1))
print('Here is the difference of both values -')
difference = y_test.squeeze() - pred
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