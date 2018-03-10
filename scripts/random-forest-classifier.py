from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np

# Loading and shuffling the data
save_dir = "../data/clf_data/sim/"
samples = np.load(save_dir+"training_data.npz")
data_2d = samples['images']
targets = samples['label']
data_2d, targets = shuffle(data_2d, targets, random_state=0)
data = np.empty([100000, 1024])

for i in range(0, 100000):
    data[i] = data_2d[i].ravel()

# Splitting the data
train_data = data[:80000]
train_targets = targets[:80000]

test_data = data[80000:]
test_targets = targets[80000:]


clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
clf.fit(train_data, train_targets)

test_pred = clf.predict(test_data)
acc = accuracy_score(test_targets, test_pred)

print(acc)