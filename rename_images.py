import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
    help="path to the directory of images")
args = vars(ap.parse_args())

fileNames = list(filter(lambda file: file[-3:] == 'jpg', os.listdir(args["path"])))
# fileNames = sorted(fileNames, key = lambda x: int(x[:-4]))
print(fileNames)

i = 1
for fileName in fileNames:
    os.rename(args["path"] + '/' + fileName, args["path"] + '/' + format(i, '07d') + '.png')
    i += 1