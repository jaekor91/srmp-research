from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import numpy as np

save_dir = "../data/clf_data/sim/"
samples = np.load(save_dir+"training_data.npz")
data_2d = samples['images']
targets = samples['label']

for estimators in [500, 1000]:
    for nodes in [8, 16, 24, 32, 40]:
        for depth in [4, 8, 12, 16, 20]:
            data_2d, targets = shuffle(data_2d, targets, random_state=0)
            data = np.empty([100000, 1024])

            for i in range(0, 100000):
                data[i] = data_2d[i].ravel()

            # Splitting the data
            train_data = data[:80000]
            train_targets = targets[:80000]

            test_data = data[80000:]
            test_targets = targets[80000:]


            clf = RandomForestClassifier(n_estimators=estimators, max_leaf_nodes=nodes, max_depth=depth, n_jobs=-1)
            clf.fit(train_data, train_targets)

            test_pred = clf.predict(test_data)
            acc = accuracy_score(test_targets, test_pred)

            print("Estimators: " + str(estimators) + " Nodes: " + str(nodes) + " Depth: " + str(depth) + " Accuracy: " + str(acc))