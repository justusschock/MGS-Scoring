import numpy as np
from PIL import Image
import os
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.cluster import spectral_clustering, KMeans
import joblib
from sklearn.feature_extraction import image
from skimage.segmentation import felzenszwalb, slic
from skimage import filters, segmentation

if sys.platform == 'linux':
    base_path = "/home/temp/schock/"
else:
    base_path = "K:/"

width = 500
height = 500

out_path = os.path.join(base_path, "MGS/KMeans")

file_list = [os.path.join(out_path, "Original_Images", x) for x in os.listdir(os.path.join(out_path, "Original_Images")) if x.endswith(".png")]
pixel_list = []

for file in file_list:
    img = Image.open(file).convert('RGB').resize((width, height))
    img_color = np.array(img)
    img_color_flattened = img_color.reshape((-1, 3))
    pixel_list += list(img_color_flattened)


for n_clusters in range(2, 16, 1):
    print("Started clustering with %02d clusters" % n_clusters)
    kmean = KMeans(verbose=False, n_clusters=n_clusters)
    kmean.fit(pixel_list)

    out_dir = os.path.join(out_path, "%d_clusters" % n_clusters)
    os.makedirs(out_dir)
    joblib.dump(kmean, os.path.join(out_dir, "kmean.pkl"))

    print("Started predicting")
    for file in file_list:
        img = Image.open(file).convert('RGB').resize((width, height))
        img_color = np.array(img)
        img_color_flattened = img_color.reshape((-1, 3))
        labels = kmean.predict(img_color_flattened)
        labels_img = labels.reshape((height, width))
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(labels_img)

        fig.savefig(os.path.join(out_dir, os.path.split(file)[-1]))
        plt.close(fig)

print("")


# img = Image.open('mouse2.png').convert('RGB').resize((width, height))
# img_color = np.array(img)
# img_color_flattened = img_color.reshape((-1, 3))
# img_red = np.array(img.split()[0])
# img_gray = np.array(img.convert('L'))

# val = filters.threshold_otsu(img_gray)
# mask = img_gray < val
# clean_border = segmentation.clear_border(mask)
#
# val_red = filters.threshold_otsu(img_red)
# mask_red = img_red < val_red
# clean_border_red = segmentation.clear_border(mask_red)
#
# clean_border_and = np.logical_and(clean_border, clean_border_red)
#
# clean_border_or = np.logical_or(clean_border, clean_border_red)
#
# fig = plt.figure(figsize=(30, 5))
# plt.subplot(1, 6, 1)
# plt.imshow(img_gray, cmap='gray')
# plt.title('Grayscale Image')
# plt.subplot(1, 6, 2)
# plt.imshow(img_red, cmap='gray')
# plt.title('Red Channel')
# plt.subplot(1, 6, 3)
# plt.imshow(clean_border, cmap='gray')
# plt.title('Otsu Thresholding on greyscale Image')
# plt.subplot(1, 6, 4)
# plt.imshow(clean_border_red, cmap='gray')
# plt.title('Otsu Thresholding on Red Channel')
# plt.subplot(1, 6, 5)
# plt.imshow(clean_border_and, cmap='gray')
# plt.title('AND connected Otsu Threshold-Masks\n on Red channel and grayscale Image')
# plt.subplot(1, 6, 6)
# plt.imshow(clean_border_or, cmap='gray')
# plt.title('OR connected Otsu Threshold-Masks\n on Red channel and grayscale Image')
#
# fig.savefig(os.path.join(out_path, "test_thresholding.png"))
# plt.close()
