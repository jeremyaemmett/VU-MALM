from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert
import imageio as iio
import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.color
import skimage.filters
import cv2

def find_nearest_white(img, target):
    nonzero = cv2.findNonZero(img)
    distances = np.sqrt((nonzero[:,:,0] - target[0]) ** 2 + (nonzero[:,:,1] - target[1]) ** 2)
    nearest_index = np.argmin(distances)
    return nonzero[nearest_index]

# read an image
image = iio.imread(r'C:\Users\Jeremy\Desktop\roots1.jpg')

# convert the image to grayscale
gray_shapes = skimage.color.rgb2gray(image)
# blur the image to denoise
blurred_shapes = skimage.filters.gaussian(gray_shapes, sigma=1.0)
# create a histogram of the blurred grayscale image
histogram, bin_edges = np.histogram(blurred_shapes, bins=256, range=(0.0, 1.0))
# create a mask based on the threshold
t = 0.8
binary_mask = blurred_shapes < t
# Invert the image
image = invert(binary_mask)
# perform skeletonization
skeleton = skeletonize(image)

skeleton = skeleton.astype(float)
binary_mask = binary_mask.astype(float)
binary_mask[binary_mask < 1.0] = 0.0

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(binary_mask, cmap=plt.cm.gray)
ax[0].imshow(skeleton, cmap=plt.cm.gray, alpha=0.3)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

median_points = cv2.findNonZero(skeleton)
print(median_points)
target = (995,1305)
for i in range(0,len(median_points),100):
    print(100.0*i/len(median_points))
    median_point = median_points[i][0]
    target = (median_point[0],median_point[1])
    #print(skeleton[target[1],target[0]])
    #print(binary_mask[target[1],target[0]])
    coords = find_nearest_white(binary_mask, target)[0]
    #print(coords,binary_mask[coords[1],coords[0]])

    ax[0].plot(target[0], target[1], color="red", marker='.')
    ax[0].plot(coords[0], coords[1], color="green", marker='.')
    ax[0].plot([target[0],coords[0]],[target[1],coords[1]],linestyle='-')

fig.tight_layout()
plt.show()