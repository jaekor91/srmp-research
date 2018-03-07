import numpy as np
import matplotlib.pyplot as plt
from resnet import ResnetBuilder
from keras import optimizers

save_dir = "../data/reg_data/"
samples = np.load(save_dir+"training_data_2.npz")


# Splitting the data
train_data = samples['images'][:40000]
train_targets = samples['mags'][:40000]

val_data = samples['images'][40000:80000]
val_targets = samples['mags'][40000:80000]

test_data = samples['images'][80000:]
test_targets = samples['mags'][80000:]


# Making the model and training
model = ResnetBuilder.build(input_shape=(1, 32, 32), num_outputs=1, block_fn='basic_block', repetitions=[15, 20, 30, 15])

model.compile(optimizer=optimizers.Adam(lr=0.00005), loss='mse', metrics=['mse'])

history = model.fit(train_data, train_targets, epochs=40, batch_size=100, validation_data=(val_data, val_targets))

model.save('magnitude_regression.h5')


# Plotting a Graph
mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mse) + 1)

plt.plot(epochs, mse, 'bo', label='Training mse')
plt.plot(epochs, val_mse, 'b', label='Validation mse')
plt.title('Training and validation mse')
plt.axis([0, 30, 0, 1])
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.axis([0, 30, 0, 10])
plt.legend()

plt.show()

# Model Summary
print(model.summary())
