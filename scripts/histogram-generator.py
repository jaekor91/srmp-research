from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load model and predict values for test set
save_dir = "../data/reg_data/"
samples_16 = np.load(save_dir+"hist_data_16.npz")
samples_17 = np.load(save_dir+"hist_data_17.npz")
samples_18 = np.load(save_dir+"hist_data_18.npz")
samples_19 = np.load(save_dir+"hist_data_19.npz")

model = load_model('magnitude_regression.h5')
samples_16_predicted = model.predict(samples_16['images'])
samples_17_predicted = model.predict(samples_17['images'])
samples_18_predicted = model.predict(samples_18['images'])
samples_19_predicted = model.predict(samples_19['images'])


# Plot histogram of the errors of the model
fig = plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.hist(samples_16_predicted - 16, bins=np.arange(-.2, .2, 0.02), histtype="step", color="black", lw=2, normed=1)
plt.title('Magnitude 16 Images')

plt.subplot(2, 2, 2)
plt.hist(samples_17_predicted - 17, bins=np.arange(-.2, .2, 0.02), histtype="step", color="black", lw=2, normed=1)
plt.title('Magnitude 17 Images')

plt.subplot(2, 2, 3)
plt.hist(samples_18_predicted - 18, bins=np.arange(-.2, .2, 0.02), histtype="step", color="black", lw=2, normed=1)
plt.title('Magnitude 18 Images')

plt.subplot(2, 2, 4)
plt.hist(samples_19_predicted - 19, bins=np.arange(-.2, .2, 0.02), histtype="step", color="black", lw=2, normed=1)
plt.title('Magnitude 19 Images')

plt.tight_layout()
plt.show()
plt.close()


print('Mag 16 std_dev:', np.std(samples_16_predicted))
print('Mag 17 std_dev:', np.std(samples_17_predicted))
print('Mag 18 std_dev:', np.std(samples_18_predicted))
print('Mag 19 std_dev:', np.std(samples_19_predicted))
