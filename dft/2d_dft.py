import random
import numpy as np
import cmath
from matplotlib import pyplot as plt
from cmath import polar
import matplotlib.gridspec as gridspec
from PIL import Image
import os
from scipy import fftpack

################################################################################
#1D discrete fourier transform

# f = [74, 22, 14, 84, 89, 40]
# x = range(0,len(f))
#
# F = np.fft.fft(f)
# k = range(0,len(F))
#
# gs = gridspec.GridSpec(3, 2)
#
# plt.subplot(gs[1:2, 0:1])
# ax = plt.gca()
# plt.scatter(x,f,c='green')
# plt.plot(x,f)
# plt.title('Original function')
# plt.xlabel('x')
# plt.ylabel('f (x)')
#
# plt.subplot(gs[0:1, 1:2])
# ax = plt.gca()
# plt.scatter(k,np.absolute(F),c='green')
# plt.plot(k,np.absolute(F))
# plt.title('Fourier coefficients - Absolute')
# plt.xlabel('k')
# plt.ylabel(r'abs ($\hat{F}$)')
#
# plt.subplot(gs[2:3, 1:2])
# ax = plt.gca()
# plt.scatter(k,np.angle(F),c='green')
# plt.plot(k,np.angle(F))
# plt.title('Fourier coefficients - Angle')
# plt.xlabel('k')
# plt.ylabel(r'ang ($\hat{F}$)')
#
# plt.show()
################################################################################

################################################################################
#2D discrete fourier transform

path = r'/home/devici/github/dft'
os.chdir(path)

f = np.asarray(Image.open('test_image1.png').convert('L'))
# f = np.asarray(Image.open('test_image2.jpeg').convert('L'))

F1 = np.fft.fft2(f)

F = fftpack.fftshift( F1 )
psd2D = np.abs( F )**2

gs = gridspec.GridSpec(3, 2)

plt.subplot(gs[1:2, 0:1])
ax = plt.gca()
plt.imshow(f,'gray')
plt.title('Original Image')

plt.subplot(gs[0:1, 1:2])
ax = plt.gca()
# plt.imshow(np.log10(np.absolute(F1)),'gray')
# plt.imshow(np.log10(np.absolute(F)),'gray')
plt.imshow(np.log10(psd2D),'gray')
plt.title('Fourier coefficients - Absolute')
# plt.colorbar()

plt.subplot(gs[2:3, 1:2])
ax = plt.gca()
plt.imshow(np.angle(F),'gray')
plt.title('Fourier coefficients - Angle')

plt.show()
