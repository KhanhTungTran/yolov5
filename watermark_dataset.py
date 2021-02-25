# import the necessary packages
from imutils import paths, resize
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import argparse
import cv2
import os
from random import seed
from random import randint, uniform
from math import ceil, floor
seed(3)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
    help="path to the watermark directory of images (assusmed to be transparent PNG)")
ap.add_argument('-i', '--input', required=True,
    help="path to the input directory of images")
ap.add_argument('-oo', '--output_original', required=True,
    help="path to the original image output directory")
ap.add_argument('-oi', '--output_image', required=True,
    help="path to the image output directory")
ap.add_argument('-ol', '--output_label', required=True,
    help="path to the label output directory")
ap.add_argument("-n", "--number", type=int, default=5000,
	help="number of images to generate")	
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())

count = 1
# TODO: loop through watermarks in watermarks directory, split for training (0.8) and testing (0.2)
for watermark_path in list(paths.list_images(args["watermark"]))[::-1][2:3]:
	print(watermark_path)
	# load the watermark image, making sure we retain the 4th channel
	# which contains the alpha transparency
	watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
	(wH, wW) = watermark.shape[:2]
	# print(watermark)
	cv2.imshow("watermark", watermark)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if args["correct"] > 0:
		(B, G, R, A) = cv2.split(watermark)
		B = cv2.bitwise_and(B, B, mask=A)
		G = cv2.bitwise_and(G, G, mask=A)
		R = cv2.bitwise_and(R, R, mask=A)
		watermark = cv2.merge([B, G, R, A])

	# not_trans_mask = watermark[:, :, 3] != 0
	# watermark[not_trans_mask] = [not_trans_mask[3], 0, 0, 0]

	# not_zero_mask = watermark[:, :, 0] != 0
	# not_255_mask =  watermark[:, :, 0] != 255
	# watermark[not_zero_mask] = [100, 100, 100, 26]
	# watermark[not_255_mask] = [200, 200, 200, 26]
	watermark[:, :, 3] = 255

	cv2.imshow("watermark", watermark)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	orig_watermark = watermark
	# cv2.imshow("watermark", watermark)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# NOTE: random input images
	list_images_path = [format(randint(1, 17125), '07d') + '.jpg' for _ in range(args["number"])]
	# loop over the input images
	for image_path in list_images_path:
		# print(image_path)
		# load the input image, then add an extra dimension to the
		# image (i.e., the alpha transparency)
		image = cv2.imread(args["input"] + '/' + image_path)
		(h, w) = image.shape[:2]
		image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
		# construct an overlay that is the same size as the input
		# image, (using an extra dimension for the alpha transparency),
		# then add the watermark to the overlay in the bottom-right
		# corner
		overlay = image.copy()

		# NOTE: random size of watermark and random location
		new_width = randint(min(int(w*3/5), 150), max(int(w*3/5), 150))
		watermark = resize(orig_watermark, width=new_width)
		(wH, wW) = watermark.shape[:2]
		while int(wH/2) >= h-int(wH/2)-1 or int(wW/2) >= w-int(wW/2)-1:
			new_width = randint(30, new_width)
			watermark = resize(orig_watermark, width=new_width)
			(wH, wW) = watermark.shape[:2]
		y_center = randint(int(wH/2), h-int(wH/2)-1)
		x_center = randint(int(wW/2), w-int(wW/2)-1)

		# not_zero_mask = watermark[:, :, 0] != 0
		# not_255_mask =  watermark[:, :, 0] != 255
		# watermark[not_zero_mask] = [0, 0, 0, 26]
		# watermark[not_255_mask] = [75, 75, 75, 26]
		overlay[y_center - floor(wH/2):y_center + ceil(wH/2), x_center - floor(wW/2):x_center + ceil(wW/2)] = watermark
		# blend the two images together using transparent overlays
		output = image.copy()

		# NOTE: Random alpha
		alpha = uniform(0.15, 0.5)
		cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)

		# not_zero_mask = watermark[:, :, 0] != 0
		# not_255_mask =  watermark[:, :, 0] != 255
		# watermark[not_zero_mask] = [125, 125, 125, 26]
		# watermark[not_255_mask] = [0, 0, 0, 26]
		# overlay[y_center - floor(wH/2):y_center + ceil(wH/2), x_center - floor(wW/2):x_center + ceil(wW/2)] = watermark
		# cv2.addWeighted(overlay, 0.5, output, 1, 0, output)

		# write the output image to disk
		image_file_name = format(count, '07d') + ".png"
		label_file_name = format(count, '07d') + ".txt"
		cv2.imwrite(os.path.sep.join((args["output_original"], image_file_name)), image)
		p = os.path.sep.join((args["output_image"], image_file_name))
		cv2.imwrite(p, output)
		# NOTE: write label to the label directory
		label = " ".join(['0', format(x_center/w, '.6f'), format(y_center/h, '.6f'), format(wW/w, '.6f'), format(wH/h, '.6f')])
		f = open(os.path.sep.join((args["output_label"], label_file_name)), 'w')
		f.write(label)
		f.close()
		count += 1