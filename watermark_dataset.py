# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--watermark', required=True,
    help="path to watermark image (assusmed to be transparent PNG)")
ap.add_argument('-i', '--input', required=True,
    help="path to the input directory of images")
ap.add_argument('-o', '--output', required=True,
    help="path to the output directory")
ap.add_argument('-a', '--alpha', type=float, default=1,
    help="alpha transparency of the overlay (smaller is more transparent)")
ap.add_argument("-c", "--correct", type=int, default=1,
	help="flag used to handle if bug is displayed or not")
args = vars(ap.parse_args())

# load the watermark image, making sure we retain the 4th channel
# which contains the alpha transparency
watermark = cv2.imread(args["watermark"], cv2.IMREAD_UNCHANGED)
(wH, wW) = watermark.shape[:2]
if args["correct"] > 0:
	(B, G, R, A) = cv2.split(watermark)
	B = cv2.bitwise_and(B, B, mask=A)
	G = cv2.bitwise_and(G, G, mask=A)
	R = cv2.bitwise_and(R, R, mask=A)
	watermark = cv2.merge([B, G, R, A])

not_trans_mask = watermark[:, :, 3] != 0
watermark[not_trans_mask] = [255, 255, 255, 255]
# watermark = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("watermark", watermark)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# loop over the input images
for imagePath in paths.list_images(args["input"]):
	# load the input image, then add an extra dimension to the
	# image (i.e., the alpha transparency)
	image = cv2.imread(imagePath)
	(h, w) = image.shape[:2]
	image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])
	# construct an overlay that is the same size as the input
	# image, (using an extra dimension for the alpha transparency),
	# then add the watermark to the overlay in the bottom-right
	# corner
	overlay = np.zeros((h, w, 4), dtype="uint8")
	overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark
	# blend the two images together using transparent overlays
	output = image.copy()
	cv2.addWeighted(overlay, args["alpha"], output, 1.0, 0, output)
	# write the output image to disk
	filename = imagePath[imagePath.rfind(os.path.sep) + 1:]
	p = os.path.sep.join((args["output"], filename))
	cv2.imwrite(p, output)