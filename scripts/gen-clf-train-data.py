from utils import *
import numpy as np
from random import randint, choice

#---- A note on conversion (David, ignore this note.)
# From Stephen: If you want to replicate the SDSS image of M2, you could use:
# 0.4 arcsec per pixel, seeing of 1.4 arcsec
# Background of 179 ADU per pixel, gain of 4.62 (so background of 179/4.62 = 38.7 photoelectrons per pixel)
# 0.00546689 nanomaggies per ADU (ie. 183 ADU = 22.5 magnitude; see mag2flux(22.5) / 0.00546689)
# Interpretation: Measurement goes like: photo-electron counts per pixel ---> ADU ---> nanomaggies.
# The first conversion is called gain.
# The second conversion is ADU to flux.

# Flux to counts conversion
# flux_to_count = 1./(ADU_to_flux * gain)

#---- Global parameters
arcsec_to_pix = 0.4
PSF_FWHM_arcsec = 1.4
PSF_FWHM_pix = PSF_FWHM_arcsec / arcsec_to_pix
gain = 4.62 # photo-electron counts to ADU
ADU_to_flux = 0.00546689 # nanomaggies per ADU
B_ADU = 179 # Background in ADU.
B_count = B_ADU/gain
flux_to_count = 1./(ADU_to_flux * gain) # Flux to count conversion

# Size of the image
num_rows = num_cols = 32 # Pixel index goes from 0 to num_rows-1


#---- Save directory
save_dir = "../data/clf_data/sim/"

#---- Number of samples for good/bad each. 
Nsample = 100 * 2

#---- Generate random magnitudes in the range 15 through 21. (samples,)
true_mags = np.random.random(size=Nsample)* 5 + 15

#---- Generate a placeholder for image tensor (samples) and labels
im_sim = np.zeros((Nsample, num_rows, num_cols, 1))# Simulated images
label = np.zeros(Nsample, dtype=bool)
label[int(Nsample/2):] = True # True is bad (because it's flagged.)

#---- Generate a mock image for each magnitude
for i, m in enumerate(true_mags):
    objs = np.array([[mag2flux(m) * flux_to_count, num_rows/2., num_cols/2.]])
    # Generate a blank image with background
    D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
    # Insert objects: For each object, evaluate the PSF located at x and y position
    f, x, y = objs[0]
    D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
    # Poission realization D of the underlying truth D0
    D = poisson_realization(D0)
    im_sim[i, :, :, 0] = D

case_list = []
for i in range(int(Nsample/2), Nsample):
    case = np.random.randint(0, 4, size=1)[0]
    case_list.append(case)

    #---- Zero out random rows and columns
    if case == 0:
        num_rows_zero_out = np.random.randint(low=1, high=4, size=1)
        num_cols_zero_out = np.random.randint(low=1, high=4, size=1)        
        random_positions = np.random.choice(range(int(num_rows/2-5), int(num_rows/2+5)), size=num_rows_zero_out, replace=False)        
        im_sim[i, random_positions, :, 0] = 0.
        random_positions = np.random.choice(range(int(num_cols/2-5), int(num_cols/2+5)), size=num_cols_zero_out, replace=False)        
        im_sim[i, :, random_positions, 0] = 0.

    #---- Zero out random block
    elif case == 1:
        corner = np.random.randint(low=num_rows/2.-4, high=num_rows/2.+4, size=2)
        im_sim[i, corner[0]-5:corner[0] + 5, corner[1]-5:corner[1] +5,] = 0.

    #---- Fudge the star
    # Generate a new image with two stars.
    elif case == 2:
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
        #--- Original
        objs = np.array([[mag2flux(true_mags[i]) * flux_to_count, num_rows/2., num_cols/2.]])
        f, x, y = objs[0]
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        #--- New
        for j in range(10):
            objs = np.array([[mag2flux(true_mags[i]) * flux_to_count * 0.9 * (np.random.random()+0.1),  num_rows/2.+np.random.randn() * 2, num_cols/2.+np.random.randn() * 2]])
            f, x, y = objs[0]
            D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)
        im_sim[i, :, :, 0] = D
        
    #---- Place the original star off the center
    elif case == 3:
        # Generate a blank image with background
        D0 = np.ones((num_rows, num_cols), dtype=float) * B_count 
        #--- Original
        objs = np.array([[mag2flux(true_mags[i]) * flux_to_count, num_rows/2., num_cols/2.]])
        f, x, y = objs[0]
        r = np.sqrt((x-num_rows/2.)**2 + (y-num_cols/2.)**2)
        while r < 3:
            x += 3 * np.random.randn()
            y += 3 * np.random.randn()
            r = np.sqrt((x-num_rows/2.)**2 + (y-num_cols/2.)**2)            
        D0 += f * gauss_PSF(num_rows, num_cols, x, y, FWHM=PSF_FWHM_pix)
        
        # Poission realization D of the underlying truth D0
        D = poisson_realization(D0)
        im_sim[i, :, :, 0] = D
        
np.savez(save_dir+"training_data.npz", images=im_sim, label=label)



# ---- View a sample of images.
# plt.close()
# fig, ax_list = plt.subplots(4, 4, figsize=(10, 10))

# for i in range(16):
#     idx_row = i // 4
#     idx_col = i % 4
#     ax_list[idx_row, idx_col].imshow(im_sim[i, :, :, 0]) #, cmap="gray")
#     ax_list[idx_row, idx_col].set_title(case_list[i])
#     ax_list[idx_row, idx_col].axis("off")

# plt.show()
# plt.close()

