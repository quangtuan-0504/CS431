from sklearn import preprocessing
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(figsize=(9, 9))

plt.subplot(231)
plt.bar(names, values)
plt.subplot(532)
plt.scatter(names, values)
plt.subplot(323)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.show()