from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
filename = raw_input("Enter the quad_clean file location :")

hdu = fits.open(filename)

h = hdu[2].data

plt.hist(h['time'], bins=np.arange(h['time'][0],h['time'][-1],5),histtype='step')
plt.show()
