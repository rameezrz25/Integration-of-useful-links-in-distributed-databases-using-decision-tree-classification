import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC  # "Support Vector Classifier"

clf = SVC(kernel='linear')

# creating datasets X containing n_samples
# Y containing two classes
X, Y = make_blobs(n_samples=500, centers=2,
                  random_state=0, cluster_std=0.40)

# plotting scatters
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');

# creating line space between -1 to 3.5
xfit = np.linspace(-1, 3.5)

# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')

# plot a line between the different sets of data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5);
x = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt',sep= '\t', header= None)
a = np.array(x)
y  = a[:,30]
x = np.column_stack((x[2],x[1]))
x.shape  # 569 samples and 2 features

print(x), (y)
clf.fit(x, y)
plt.show()
