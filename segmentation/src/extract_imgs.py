import os
import cv2
import numpy as np


edge_upper_left = (0, 0)
edge_middle = (0, 0)
edge_bottom_right = (0, 0)

file_name = "/home/temp/schock/MGS/MVI_0315.MOV"
out_dir = "/home/temp/schock/MGS/" + os.path.split(file_name)[-1].rsplit(".")[0]

frames = []

os.makedirs(os.path.join(out_dir, "whole_imgs"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "separated_imgs"), exist_ok=True)
vidcap = cv2.VideoCapture(file_name)
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    # if count == 0:
    #   tmp = input("Please enter coordinates of upper left edge (separated by \",\"):").split(",")
    #   edge_upper_left = (int(tmp[0]), int(tmp[1]))
    #   tmp = input("Please enter coordinates of middle edge (separated by \",\"):").split(",")
    #   edge_middle = (int(tmp[0]), int(tmp[1]))
    #   tmp = input("Please enter coordinates of bottom right edge (separated by \",\"):").split(",")
    #   edge_bottom_right = (int(tmp[0]), int(tmp[1]))

    cv2.imwrite(os.path.join(out_dir, "frame%d.jpg" % count), image)     # save frame as JPEG file
    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
        break
    count += 1
