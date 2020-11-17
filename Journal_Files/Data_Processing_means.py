import pandas as pd
import matplotlib.pyplot as plt
import addcopyfighandler

folder = 'angle_v_1.csv'

dataset = pd.read_csv('Data_Xihan_11-5_components/' + folder)
#added condition to clean up the data
#thanks to https://stackoverflow.com/questions/49546428
dataset = dataset.iloc[:, 1:3]

#added mean condition thanks to
#https://stackoverflow.com/questions/64879466
means = dataset.groupby('val').mean()
print(means.shape)
print(means)

plt.plot(means)
plt.xlabel('Value')
plt.ylabel('Angle')
plt.title('Output plot for ' + folder)
plt.show()