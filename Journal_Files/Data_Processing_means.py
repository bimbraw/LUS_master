import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler

folder = 'angle_v_1'
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
print(means)

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