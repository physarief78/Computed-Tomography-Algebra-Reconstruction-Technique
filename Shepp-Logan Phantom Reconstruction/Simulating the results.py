import numpy as np
import matplotlib.pyplot as plt

# Load the text files
phantom = np.loadtxt('shepp_logan_phantom.txt')
sinogram = np.loadtxt('sinogram_sl.txt')
reconstruction = np.loadtxt('reconstruction_tv_sl.txt')

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(phantom, cmap='grey')
ax[0].set_title('Original Phantom', fontname="Times New Roman", fontweight="bold", )

ax[1].imshow(sinogram, cmap='grey', aspect='auto')
ax[1].set_title('Sinogram (Radon Transform)', fontname="Times New Roman", fontweight="bold", fontsize='20')

ax[2].imshow(reconstruction, cmap='grey')
ax[2].set_title('Reconstructed Image (ART)', fontname="Times New Roman", fontweight="bold", fontsize='20')

plt.show()
