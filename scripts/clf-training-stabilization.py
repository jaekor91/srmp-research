import numpy as np
import matplotlib.pyplot as plt
from resnet_clf import ResnetBuilder
from keras import optimizers
from keras import losses
from sklearn.utils import shuffle
import pickle

save_dir = "../data/clf_data/sim/"
samples = np.load(save_dir + "training_data.npz")
data = samples['images']
targets = samples['label']

for i in range(0, 5):
    data, targets = shuffle(data, targets, random_state=0)

    train_data = data[:40000]
    train_targets = targets[:40000]

    val_data = data[40000:80000]
    val_targets = targets[40000:80000]

    test_data = data[80000:]
    test_targets = targets[80000:]

    model = ResnetBuilder.build(input_shape=(1, 32, 32), num_outputs=1, block_fn='basic_block',
                                repetitions=[3, 4, 6, 3])
    model.compile(optimizer=optimizers.RMSprop(lr=0.00001), loss=losses.binary_crossentropy,
                  metrics=['binary_crossentropy', 'accuracy'])

    history = model.fit(train_data, train_targets, epochs=40, batch_size=80, validation_data=(val_data, val_targets))

    pickle.dump(history.history, open(save_dir + "history.p" + str(i), "wb"))
    del model

# Plotting a Graph
fig = plt.figure(figsize=(10, 10))
for i in range(0, 5):
    history = pickle.load(open(save_dir + "history.p" + str(i), "rb"))
    bce = history['binary_crossentropy']
    val_bce = history['val_binary_crossentropy']
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['acc']
    val_acc = history['val_acc']

    epochs = range(1, len(bce) + 1)

    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss, 'b', label='Trial' + str(i + 1))
    plt.title('Training Loss')
    plt.axis([0, 40, 0, 1])
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_loss, 'b', label='Trial' + str(i + 1))
    plt.title('Validation Loss')
    plt.axis([0, 40, 0, 1])
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, acc, 'b', label='Trial' + str(i + 1))
    plt.title('Training Accuracy')
    plt.axis([0, 40, .99, 1])
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_acc, 'b', label='Trial' + str(i + 1))
    plt.title('Validation Accuracy')
    plt.axis([0, 40, .99, 1])
    plt.legend()

plt.show()