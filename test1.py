# Image from LIDC-IDRI image database

#from pathlib import Pathpip
import pydicom
import matplotlib.pyplot as plt

#packages needed for thresholding algorithm 
import glob
import imageio.v3 as iio
import ipympl
import numpy as np
import skimage as ski

from skimage.transform import resize 
import cv2
import os 

import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5 backend 

dicom_file = pydicom.read_file('/Users/vrundapatel/Desktop/DMIC/manifest-1713483093100/LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-26851/3000972.000000-NA-22902/1-1.dcm')

# information from the dicom file
#print(dicom_file)

#number of rows in dicom file 
print('number rows in DICOM file:', dicom_file.Rows)

# extract image 
ct = dicom_file.pixel_array
fig = plt.figure()
fig.suptitle("Original CT Scan")
plt.imshow(ct, cmap = 'gray')
#plt.imshow(ct)
plt.show()


#blur the image to denoise 
blurred_shapes = ski.filters.gaussian(ct, sigma = 1.0)
fig, ax = plt.subplots()
fig.suptitle("Blurred CT Scan (to denoise image)")
ax.imshow(blurred_shapes, cmap='gray')

# finding exact bin size
pixel_array = dicom_file.pixel_array
height, width = pixel_array.shape
pixels = height * width
b = int(np.sqrt(pixels))
print("number bins:", b)

# histogram of the blurred grayscale image 
histogram, bin_edges = np.histogram(blurred_shapes, bins = b, range=(0.0,1.0))
# usually gray scale values go between 0 and 1
fig, ax = plt.subplots()
ax.plot(bin_edges[0:-1], histogram)
ax.set_title("Grayscale Histogram")
ax.set_xlabel("grayscale value")
ax.set_ylabel('pixels')
ax.set_xlim(0.0, 0.3)
plt.show()

# input image needs to be 8 or 16 bit single channel image
blurred_shapes = cv2.convertScaleAbs(blurred_shapes)
# Otsu method for findin treshold value
otsu_threshold, image_result = cv2.threshold(blurred_shapes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Obtained Threshold: ", otsu_threshold)


# turning pixels "off"
# create a mask based on the threshold 
# average of the two grayscale values were used as threshold - article
t = otsu_threshold
binary_mask = (blurred_shapes < 0.02)
fig, ax = plt.subplots()
ax.imshow(binary_mask, cmap='gray')
plt.show()
# the part it identified is in white 


#intensity of a pixel 
directory = r'/Users/vrundapatel/Desktop/DMIC'
imagedata = pydicom.dcmread('/Users/vrundapatel/Desktop/DMIC/manifest-1713483093100/LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-26851/3000972.000000-NA-22902/1-1.dcm')
os.chdir(directory)


print("Before saving image")
print(os.listdir(directory))
img = imagedata.pixel_array
#print(img)
img = resize(img, (512, 512))

# saving resized image 
cv2.imwrite('SegmentedCTscan.png', img)

print("After saving image")
print(os.listdir(directory))
print("Successfully saved")
