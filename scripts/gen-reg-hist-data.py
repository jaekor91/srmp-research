from utils import *

# ---- A note on conversion (David, ignore this note.)
# From Stephen: If you want to replicate the SDSS image of M2, you could use:
# 0.4 arcsec per pixel, seeing of 1.4 arcsec
# Background of 179 ADU per pixel, gain of 4.62 (so background of 179/4.62 = 38.7 photoelectrons per pixel)
# 0.00546689 nanomaggies per ADU (ie. 183 ADU = 22.5 magnitude; see mag2flux(22.5) / 0.00546689)
# Interpretation: Measurement goes like: photo-electron counts per pixel ---> ADU ---> nanomaggies.
# The first conversion is called gain.
# The second conversion is ADU to flux.

# Flux to counts conversion
# flux_to_count = 1./(ADU_to_flux * gain)

# ---- Global parameters
arcsec_to_pix = 0.4
PSF_FWHM_arcsec = 1.4
PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix
gain = 4.62  # photo-electron counts to ADU
ADU_to_flux = 0.00546689  # nanomaggies per ADU
B_ADU = 179  # Background in ADU.
B_count = B_ADU / gain
flux_to_count = 1. / (ADU_to_flux * gain)  # Flux to count conversion

# Size of the image
num_rows = num_cols = 32  # Pixel index goes from 0 to num_rows-1

# # Prior parameters
# alpha = -1.1# f**alpha
# fmin = mag2flux(20) # Minimum flux of an object.
# fmax = 17. # Maximum flux in the simulation

# ---- Save directory
save_dir = "../data/reg_data/"

# ---- Number of samples
Nsample = 10000

for mag in [16, 17, 18, 19]:
    # ---- Generate magnitudes of a single number. (samples, magnitude)
    true_mags = np.full(Nsample, mag)

    # ---- Generate a placeholder for image tensor (samples)
    im_sim = np.zeros((Nsample, num_rows, num_cols, 1))  # Simulated images

    # ---- Generate a mock image for each magnitude
    for i, m in enumerate(true_mags):
        objs = np.array([[mag2flux(m) * flux_to_count, num_rows / 2., num_cols / 2.]])
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count
        # Insert objects: For each object, evaluate the PSF located at x and y position
        f, x, y = objs[0]
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)
        im_sim[i, :, :, 0] = D

    np.savez(save_dir + "hist_data_" + str(mag) + ".npz", images=im_sim, mags=true_mags)
