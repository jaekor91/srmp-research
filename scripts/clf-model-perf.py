from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

save_dir = "../data/clf_data/sim/"

model = load_model('classification_model.h5')

samples = np.load(save_dir+"training_data1.npz")
data = samples['images']
targets = samples['label']
contam = samples['contamination']

predictions_temp = model.predict(data)
predictions = predictions_temp > .5

for i in range(0, len(data)):
    if predictions[i] != targets[i]:
        temp = data[i]
        img = np.empty((32, 32))
        for j in range(0, 32):
            for k in range(0, 32):
                img[j][k] = temp[j][k][0]

        if contam[i] == 0:
            print('No contamination: ' + str(i))
        elif contam[i] == 1:
            print('Rows and columns zerod out: ' + str(i))
        elif contam[i] == 2:
            print('Random block zerod out: ' + str(i))
        elif contam[i] == 3:
            print('Star is fudged: ' + str(i))
        elif contam[i] == 4:
            print('Star is off center: ' + str(i))

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.savefig(save_dir + 'test_' + str(i) + '.png')
        plt.show()