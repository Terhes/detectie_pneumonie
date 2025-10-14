# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# === Load the image ===
path = r'C:\Users\andreit\Downloads\person100_bacteria_478.jpeg'
test2 = plt.imread(path)

# === Transformarea unei imagini color intr-o imagine alb-negru ===
def rgb2gray(img_in, tip='JPEG'):
    s = img_in.shape
    if len(s) == 3 and s[2] >= 3:
        gray = 0.299 * img_in[:, :, 0] + 0.587 * img_in[:, :, 1] + 0.114 * img_in[:, :, 2]
        if tip.lower() == 'png':
            gray = gray * 255  # PNGs are often normalized [0,1]
        return gray
    else:
        print('The image is not a color image')
        return img_in  # fallback

def negativare(img_in):
    return 255 - img_in

def binarizare(img_in, threshold):
    img_out = np.zeros_like(img_in)
    img_out[img_in >= threshold] = 255
    return img_out

# === Process images ===
img_out = rgb2gray(test2, 'JPEG')
img_out = img_out.astype('uint8')

plt.figure()
plt.imshow(img_out, cmap='gray')
plt.title("Original Image")

img_negativata = negativare(img_out)
plt.figure()
plt.imshow(img_negativata, cmap='gray')
plt.title("Negative Image")

img_binarizata = binarizare(img_out, 190)
plt.figure()
plt.imshow(img_binarizata, cmap='gray')
plt.title("Binarized Image")

# === Histograms ===
plt.figure(figsize=(15, 5))

# Histogram 1: Gray image
plt.subplot(1, 3, 1)
plt.hist(img_out.ravel(), bins=256, range=(0, 255))
plt.title("Histogram - Original Image")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

# Histogram 2: Negative image
plt.subplot(1, 3, 2)
plt.hist(img_negativata.ravel(), bins=256, range=(0, 255))
plt.title("Histogram - Negative Image")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

# Histogram 3: Binarized image
plt.subplot(1, 3, 3)
plt.hist(img_binarizata.ravel(), bins=256, range=(0, 255))
plt.title("Histogram - Binarized Image")
plt.xlabel("Pixel intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
