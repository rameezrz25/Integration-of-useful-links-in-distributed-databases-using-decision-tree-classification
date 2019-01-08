# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
#matplotlib inline

# Pickle package
import pickle
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt',sep= '\t', header= None)
print(df)
# Construct iris plot
sns.swarmplot(x=0, y=5, data=df)

# Show plot
plt.show()
