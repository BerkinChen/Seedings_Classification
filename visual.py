import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)

data = []
with open('result.txt','r') as f:
    for line in f.readlines():
        data.append(float(line.split(' ')[-1]))
data = np.array(data).reshape(4,3,3,3)
data = np.transpose(data,(2,3,1,0))

for i, row in enumerate(axes):
    for j, col in enumerate(row):
        print(data[i][j][0])
        print(data[i][j][1])
        print(data[i][j][2])
        col.plot([0.3, 1, 1.7,2.4], data[i][j][0], color='red',
                 linewidth=0.5, marker='o', markersize=4,alpha=0.6)
        col.plot([0.3, 1, 1.7,2.4], data[i][j][1], color='blue',
                 linewidth=0.5, marker='s', markersize=4, alpha=0.6)
        col.plot([0.3, 1, 1.7, 2.4], data[i][j][2],
                 color='green', linewidth=0.5, marker='x', markersize=4, alpha=0.6)
        col.set_xticks([0.3, 1, 1.7, 2.4], [
                       'None', 'Normalize', 'RandomFlip', 'RandomCrop'],fontsize=8)
        col.set_xlim(0, 2.7)
axes[0, 0].set_ylabel('None', fontsize=18)
axes[1, 0].set_ylabel('Dropout', fontsize=18)
axes[2, 0].set_ylabel('Weightdecay', fontsize=18)
axes[2, 0].set_xlabel('Adam', fontsize=18)
axes[2, 1].set_xlabel('SGD', fontsize=18)
axes[2, 2].set_xlabel('Adagrad', fontsize=18)
fig.legend(['MLP', 'Resnet','Vgg'],fontsize=14)
fig.suptitle('Results',fontsize=36)

plt.savefig('result.png')

