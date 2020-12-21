import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

folder = 'angle_v_2'
xlabel_u = 'U values'
xlabel_v = 'V values'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder + '.csv')
print(dataset)
print(dataset.shape)
X = dataset.iloc[:, 2].values
X = X.reshape(len(X), 1)
y = dataset.iloc[:, 1].values

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

print('These are the means -')
print(means)

#need to recheck this - some difference one value missing
print('Val -')
m = []
for i in range(66):
    m.append(i+122)
print(f'm = {m}')

#test training split
X_train, X_test, y_train, y_test = train_test_split(m, means, test_size=0.2, random_state=42)

print('Training set - X')
print(X_train)
print('Test set - X')
print(X_test)
print('Training set - y')
print(y_train)
print('Test set - y')
print(y_test)

mse_Test = metrics.mean_squared_error(y_test, X_test)
print('Mean squared error for the test set -')
print(mse_Test)
mse_Train = metrics.mean_squared_error(y_train, X_train)
print('Mean squared error for the training set -')
print(mse_Train)

fig = plt.figure()
plt.scatter(X, y, color='pink', label='data')
plt.plot(X_compressed, means)
plt.xlabel(xlabel_v)
plt.ylabel('Angle')
plt.xlim(0, 255)
plt.ylim(-50, 50)
plt.title('Output plot for ' + folder)
plt.savefig('C:/Users/bimbr/OneDrive/Desktop/Research/Lung_Ultrasound/Updated_plots/' + folder + '.png')
#plt.show()