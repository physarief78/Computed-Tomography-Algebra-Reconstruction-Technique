import numpy as np
import matplotlib.pyplot as plt

# Load the text files
phantom = np.loadtxt('phantom.txt')
sinogram = np.loadtxt('sinogram.txt')
reconstruction = np.loadtxt('reconstruction_tv.txt')

# Plot results
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(phantom, cmap='inferno')
ax[0].set_title('Original Phantom')

ax[1].imshow(sinogram, cmap='inferno', aspect='auto')
ax[1].set_title('Sinogram (Radon Transform)')

ax[2].imshow(reconstruction, cmap='inferno')
ax[2].set_title('Reconstructed Image (ART)')

plt.show()




















