import numpy as np
import matplotlib.pyplot as plt
from resnet import ResnetBuilder

save_dir = "../data/reg_data/"
samples = np.load(save_dir+"training_data.npz")


# Splitting the data
train_data = samples['images'][:20000]
train_targets = samples['mags'][:20000]

val_data = samples['images'][20000:40000]
val_targets = samples['mags'][20000:40000]

test_data = samples['images'][40000:]
test_targets = samples['mags'][40000:]


# Making the model and training
model = ResnetBuilder.build(input_shape=(1, 32, 32), num_outputs=1, block_fn='basic_block', repetitions=[3, 4, 6, 3])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc', 'mse'])

history = model.fit(train_data, train_targets, epochs=30, batch_size=40, validation_data=(val_data, val_targets))

model.save('magnitude_regression_1.h5')


# Plotting a graph
mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mse) + 1)

plt.plot(epochs, mse, 'bo', label='Training mse')
plt.plot(epochs, val_mse, 'b', label='Validation mse')
plt.title('Training and validation mse')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




