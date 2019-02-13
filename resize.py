import utils
import os
import cv2 as cv

count = 0
for filename in os.listdir(utils.DATA_DIR_IMG):
    if filename == 'Premature end of JPEG file':
        print('INCORRECT')
    if filename is not None:
        print(filename)
        count = count + 1
        oriimg = cv.imread(utils.DATA_DIR_IMG + filename)
        if oriimg is None:
            print('AILE')
print(count)