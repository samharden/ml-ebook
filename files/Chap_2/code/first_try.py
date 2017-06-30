import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

data = load_iris()
features = data['data']
feature_names = data['feature_names']
target = data['target']
labels = data['target_names'][data['target']]

# for t, marker, c in zip(range(3), ">ox", "rbg"):
#     plt.scatter(
#     features[target == t,0],
#     features[target == t,1],
#     marker=marker,
#     c=c)
#     plt.show()

plength = features[:, 2]
is_setosa = (labels=='setosa')
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()

print('Max of setosa: {0}.'.format(max_setosa))
print('Min of setosa: {0}.'.format(min_non_setosa))
print('Maximum of setosa: {0}.'.format(plength[is_setosa].max()))
print('Minimum of others: {0}.'.format(plength[~is_setosa].min()))
