import numpy as np
import matplotlib.pyplot as plt
from resnet_clf import ResnetBuilder
from keras import optimizers
from keras import losses
from sklearn.utils import shuffle

# Loading and shuffling the data
save_dir = "../data/clf_data/sim/"
samples = np.load(save_dir+"training_data.npz")
data = samples['images']
targets = samples['label']
data, targets = shuffle(data, targets, random_state=0)

# Splitting the data
train_data = data[:40000]
train_targets = targets[:40000]

val_data = data[40000:80000]
val_targets = targets[40000:80000]

test_data = data[80000:]
test_targets = targets[80000:]

# Making the model and training
model = ResnetBuilder.build(input_shape=(1, 32, 32), num_outputs=1, block_fn='basic_block', repetitions=[3, 4, 6, 3])

model.compile(optimizer=optimizers.RMSprop(lr=0.00005), loss=losses.binary_crossentropy, metrics=['binary_crossentropy', 'accuracy'])

history = model.fit(train_data, train_targets, epochs=25, batch_size=80, validation_data=(val_data, val_targets))

model.save('classification_model.h5')


# Plotting a Graph
bce = history.history['binary_crossentropy']
val_bce = history.history['val_binary_crossentropy']
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1, len(bce) + 1)

plt.plot(epochs, bce, 'bo', label='Training Cross Entropy')
plt.plot(epochs, val_bce, 'b', label='Validation Cross Entropy')
plt.title('Training and Validation Cross Entropy')
plt.axis([0, 25, 0, .05])
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.axis([0, 25, 0, 1])
plt.legend()

plt.figure()

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.axis([0, 25, .99, 1])
plt.legend()

plt.show()


# Model Summary
print(model.summary())
