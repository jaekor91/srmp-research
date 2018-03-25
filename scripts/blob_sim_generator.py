from utils import *


# Basic parameters
# - Total flux could range 2,500 to 25,000
# - Diagonal 2D covariance 16 = r^2 of varying from 0.1 to 0.9 randomly
fmax = 25000
fmin = 20000
nrows = ncols = 32 # 
B = 200 # Background per pixel
FWHM_min = 5
FWHM_max = 8
rho_min = -0.8
rho_max = 0.8
scatter_max = 1
num_comps_max = 10
# For double peaks only
sep_peaks_min = 5
sep_peaks_max = 8



# --- Place holder for images and label
Nsample = 25
im_arr = np.zeros((Nsample, nrows, ncols))
label_arr = np.zeros(Nsample, dtype=int)
# 0: background; 1: single; 2: double; Network would first determine 
# if there is a peak and then further try to distinguish it if it's single or double peak.



# --- Generate image
for i in xrange(Nsample):
    im = np.ones((nrows, ncols)) * B # Background

    # Whether to generate a single peak (r < 0.5) or double peaks (r>0.5)
    r = np.random.random()
    if r < 1/3.: # Single
        label = 1
        y = np.random.randint(2, ncols-2, 1)[0] # Wavelength position
        f = (fmax - fmin) * np.random.random() + fmin # Random flux selection
        rho = (rho_max - rho_min) * np.random.random() + rho_min # Covarince selection
        FWHM = (FWHM_max - FWHM_min) * np.random.random() + FWHM_min # FWHM selection
        num_comps = np.random.randint(1, num_comps_max, 1)[0]
        scatter = np.random.random() * scatter_max
        peak = f * generalized_gauss_PSF(nrows, ncols, nrows//2,  y, FWHM=FWHM, rho=rho, scatter=scatter, num_comps=num_comps) # Double peak
        im += peak # Add peak
    elif r > 2/3.: # Double
        label = 2
        y = np.random.randint(5, ncols-5, 1)[0] # Wavelength position of the center
        sep_peaks = np.random.random() * (sep_peaks_max - sep_peaks_min) + sep_peaks_min # Separation in peaks in pixels
        f = (fmax - fmin) * np.random.random() + fmin # Random flux selection
        rho = (rho_max - rho_min) * np.random.random() + rho_min # Covarince selection
        FWHM = (FWHM_max - FWHM_min) * np.random.random() + FWHM_min # FWHM selection
        num_comps = np.random.randint(1, num_comps_max, 1)[0]
        scatter = np.random.random() * scatter_max
        peak1 = f * generalized_gauss_PSF(nrows, ncols, nrows//2,  y-sep_peaks/2., FWHM=FWHM, rho=rho, scatter=scatter, num_comps=num_comps) # Double peak
        peak2 = f * generalized_gauss_PSF(nrows, ncols, nrows//2,  y+sep_peaks/2., FWHM=FWHM, rho=rho, scatter=scatter, num_comps=num_comps) # Double peak    
        im += (peak1 + peak2) # Add peak
    else:
        label = 0
        pass

    im = poisson_realization(im) # Poisson realization
    im -= B # Background subtraction    
    
    # -- Save the image and its label
    im_arr[i] = im
    label_arr[i] = label
    
    
# ---- Save the training data
np.savez("../data/blob_data/blob_training_data.npz", image=im_arr, label=label_arr)

# ---- Load image
data = np.load("../data/blob_data/blob_training_data.npz")
im_arr= data["image"]
label_arr = data["label"]

    
# ---- View a sample of images.
plt.close()
fig, ax_list = plt.subplots(5, 5, figsize=(10, 10))

i_start = 0
i_end = i_start + 25
for i in range(i_start, i_end):
    idx_row = (i-i_start) // 5
    idx_col = (i-i_start) % 5
    ax_list[idx_row, idx_col].imshow(im_arr[i, :, :], cmap="gray", interpolation="none") # , vmin=vmin, vmax=vmax)
    ax_list[idx_row, idx_col].set_title(label_arr[i])
    ax_list[idx_row, idx_col].axis("off")    

plt.show()
plt.close()
