# import the necessary packages
from imutils import paths, resize
import numpy as np
import argparse
import cv2
import os
from random import seed
from random import randint, uniform

seed(1)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
    help="path to the watermark directory of images (assusmed to be transparent PNG)")
ap.add_argument('-i', '--input', required=True,
    help="path to the input directory of images")
ap.add_argument('-oi', '--output_image', required=True,
    help="path to the image output directory")
ap.add_argument('-ol', '--output_label', required=True,
    help="path to the label output directory")
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())

# TODO: loop through watermarks in watermarks directory, split for training (0.8) and testing (0.2)
# load the watermark image, making sure we retain the 4th channel
# which contains the alpha transparency
watermark = cv2.imread(args["watermark"], cv2.IMREAD_UNCHANGED)
(wH, wW) = watermark.shape[:2]

cv2.imshow("watermark", watermark)
cv2.waitKey(0)
cv2.destroyAllWindows()
if args["correct"] > 0:
	(B, G, R, A) = cv2.split(watermark)
	B = cv2.bitwise_and(B, B, mask=A)
	G = cv2.bitwise_and(G, G, mask=A)
	R = cv2.bitwise_and(R, R, mask=A)
	watermark = cv2.merge([B, G, R, A])

not_trans_mask = watermark[:, :, 3] != 0
watermark[not_trans_mask] = [255, 255, 255, 255]

# TODO: random input images
# loop over the input images
for image_path in paths.list_images(args["input"]):
	# load the input image, then add an extra dimension to the
	# image (i.e., the alpha transparency)
	image = cv2.imread(image_path)
	(h, w) = image.shape[:2]
	image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
	# construct an overlay that is the same size as the input
	# image, (using an extra dimension for the alpha transparency),
	# then add the watermark to the overlay in the bottom-right
	# corner
	overlay = np.zeros((h, w, 4), dtype="uint8")

    # NOTE: random size of watermark and random location
	new_width = randint(20, 200)
	watermark = resize(watermark, new_width=100)
	(wH, wW) = watermark.shape[:2]
	y_center = randint(int(wH/2), h-int(wH/2)-1), x_center = randint(int(wW/2), h-int(wW/2)-1)
	overlay[y_center - int(wH/2):y_center + int(wH/2), x_center - int(wW/2):x_center + int(wW/2)] = watermark
	# blend the two images together using transparent overlays
	output = image.copy()

    # NOTE: Random alpha
	alpha = uniform(0.1, 0.9)
	cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)

	# write the output image to disk
	file_name = image_path[image_path.rfind(os.path.sep) + 1:]
	p = os.path.sep.join((args["output_image"], file_name))
	cv2.imwrite(p, output)
    # NOTE: write label to the label directory
	label = " ".join('0', format(x_center/w), format(wW/w), format(y_center/h), format(wH/h))