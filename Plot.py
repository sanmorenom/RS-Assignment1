import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.signal import savgol_filter
from ast import literal_eval
x_list = []
titles = []
folder = f'CSVs/'
files = os.listdir(folder)
index = 0
print('start checking files')
while index < len(files):
    x = []
    filename = files[index]
    if filename.endswith('.csv'):
        with open(f'{folder}{filename}','r') as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            for row in lines:
                
                if row[1] == 'val_error':
                    titles.append(f'Model {index+1}')
                elif row[0] == '0':
                    x.append(literal_eval(row[1]))
                    
        x_list.append(x)

    index +=1

#learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing

fig, ax = plt.subplots()
ax.set_title(f'Experiment: Validation loss over epochs for the top 5 models.')
ax1 = ax.twinx()
leg = []

print(x_list[0][0])
for i in range(len(x_list)):

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    lns1 = ax.plot(range(len(x_list[i][0])), x_list[i][0], label = f'{titles[i]}')
    leg += lns1 

labs = [l.get_label() for l in leg]    
ax.legend(leg, labs, loc=0)    
plt.savefig(f"result.png", dpi=300, bbox_inches='tight')
plt.show()