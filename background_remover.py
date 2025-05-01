import pandas as pd
import numpy as np
import cv2 


img=cv2.imread("white_bg\cataract\_130_3561448.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

img[thresh == 255] = 0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
erosion = cv2.erode(img, kernel, iterations = 1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow("image", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()










"""def bg_remove(path):
    img=cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    img[thresh == 255] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(img, kernel, iterations = 1)
    return erosion

import os
input='C:/Users/ASUS/Desktop/Newfolder/normal'
output='C:/Users/ASUS/Desktop/Outfolder'
for image in os.listdir(input):
    input_image_path=os.path.join(input,image)
    outimg=bg_remove(input_image_path)
    output_image_path=os.path.join(output,image)
    cv2.imwrite(output_image_path,outimg)"""