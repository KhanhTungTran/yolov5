# import the necessary packages
from imutils import paths, resize
import numpy as np
import argparse
import cv2
import os
from random import seed
from random import randint, uniform
from math import ceil, floor
seed(1)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
    help="path to the watermark directory of images (assusmed to be transparent PNG)")
ap.add_argument('-i', '--input', required=True,
    help="path to the input directory of images")
ap.add_argument('-oo', '--output_original', required=True,
    help="path to the original iamge output directory")
ap.add_argument('-oi', '--output_image', required=True,
    help="path to the image output directory")
ap.add_argument('-ol', '--output_label', required=True,
    help="path to the label output directory")
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())

count = 28501
# TODO: loop through watermarks in watermarks directory, split for training (0.8) and testing (0.2)
for watermark_path in list(paths.list_images(args["watermark"]))[38:]:
	print(watermark_path)
	# load the watermark image, making sure we retain the 4th channel
	# which contains the alpha transparency
	watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
	(wH, wW) = watermark.shape[:2]

	if args["correct"] > 0:
		(B, G, R, A) = cv2.split(watermark)
		B = cv2.bitwise_and(B, B, mask=A)
		G = cv2.bitwise_and(G, G, mask=A)
		R = cv2.bitwise_and(R, R, mask=A)
		watermark = cv2.merge([B, G, R, A])

	not_trans_mask = watermark[:, :, 3] != 0
	watermark[not_trans_mask] = [255, 255, 255, 255]
	orig_watermark = watermark
	# cv2.imshow("watermark", watermark)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# NOTE: random input images
	list_images_path = [format(randint(1, 17125), '07d') + '.jpg' for _ in range(750)]
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
		overlay = np.zeros((h, w, 4), dtype="uint8")

		# NOTE: random size of watermark and random location
		new_width = randint(30, int(w*3/5))
		watermark = resize(orig_watermark, width=new_width)
		(wH, wW) = watermark.shape[:2]
		while int(wH/2) >= h-int(wH/2)-1 or int(wW/2) >= w-int(wW/2)-1:
			new_width = randint(30, new_width)
			watermark = resize(orig_watermark, width=new_width)
			(wH, wW) = watermark.shape[:2]
		y_center = randint(int(wH/2), h-int(wH/2)-1)
		x_center = randint(int(wW/2), w-int(wW/2)-1)
		overlay[y_center - floor(wH/2):y_center + ceil(wH/2), x_center - floor(wW/2):x_center + ceil(wW/2)] = watermark
		# blend the two images together using transparent overlays
		output = image.copy()

		# NOTE: Random alpha
		alpha = uniform(0.1, 0.9)
		cv2.addWeighted(overlay, alpha, output, 1.0, 0, output)

		# write the output image to disk
		image_file_name = format(count, '07d') + ".png"
		label_file_name = format(count, '07d') + ".txt"
		cv2.imwrite(os.path.sep.join((args["output_original"], image_file_name)), image)
		p = os.path.sep.join((args["output_image"], image_file_name))
		cv2.imwrite(p, output)
		# NOTE: write label to the label directory
		label = " ".join(['0', format(x_center/w, '.6f'), format(wW/w, '.6f'), format(y_center/h, '.6f'), format(wH/h, '.6f')])
		f = open(os.path.sep.join((args["output_label"], label_file_name)), 'w')
		f.write(label)
		f.close()
		count += 1