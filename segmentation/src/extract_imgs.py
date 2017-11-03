import os
import cv2
import numpy as np
import imageio
from multiprocessing import Pool
from functools import partial
import json
from PIL import Image


def extract_frames(file_path, base_path):
    path, folder = os.path.split(file_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])
    os.makedirs(out_path, exist_ok=True)

    print("Extracting frames from \n %s \n \t to \n %s" % (file_path, out_path))

    reader = imageio.get_reader(file_path)
    for i, im in enumerate(reader):
        imageio.imwrite(os.path.join(out_path, "frame_%d.png" % i), im)

    print("Finished Extracting %s" % file_path)


def get_vid_points(vid_path, base_path):
    path, folder = os.path.split(vid_path)
    out_path = os.path.join(base_path, os.path.split(path)[-1], folder.rsplit(".", maxsplit=1)[0])

    if not os.path.exists(out_path):
        extract_frames(vid_path, base_path)

    image = imageio.imread(os.path.join(out_path, "frame_1.png"))
    image.view()

    upper_left_x = int(input("X-Coordinate of upper left corner:"))
    upper_left_y = int(input("Y-Coordinate of upper left corner:"))
    middle_x = int(input("X-Coordinate of middle point:"))
    middle_y = int(input("Y-Coordinate of middle point:"))
    bottom_right_x = int(input("X-Coordinate of bottom right corner:"))
    bottom_right_y = int(input("Y-Coordinate of bottom right corner:"))

    tmp_dict = {'Upper Left X': upper_left_x,
                'Upper Left Y': upper_left_y,
                'Middle X': middle_x,
                'Middle Y': middle_y,
                'Bottom Right X': bottom_right_x,
                'Bottom Right Y': bottom_right_y}

    return tmp_dict


def extract_boxes_from_frames(selected_movie, base_dir):
    current_dict = movie_dict[selected_movie]

    upper_left_box_coordinates = ((current_dict["Upper Left X"], (current_dict["Middle X"]),
                                   current_dict["Upper Left Y"], current_dict["Middle Y"]))

    upper_right_box_coordinates = ((current_dict["Middle X"], current_dict["Bottom Right X"]),
                                   (current_dict["Upper Left Y"], current_dict["Middle Y"]))

    bottom_left_box_coordinates = ((current_dict["Upper Left X"], current_dict["Middle X"]),
                                   (current_dict["Middle Y"], current_dict["Bottom Right Y"]))

    bottom_right_box_coordinates = ((current_dict["Middle X"], current_dict["Bottom Right X"]),
                                    (current_dict["Middle Y"], current_dict["Bottom Right Y"]))

    movie_dir = os.path.join(base_dir, selected_movie)

    frames = [os.path.join(movie_dir, x) for x in os.listdir(movie_dir) if (x.endswith(".png") and
                                                                            os.path.isfile(os.path.join(movie_dir, x)))]
    frames.sort()

    separated_out_dir = os.path.join(os.path.split(base_dir)[0], "Separated", selected_movie)
    os.makedirs(separated_out_dir, exist_ok=True)

    for frame in frames:
        img = np.array(Image.open(frame).convert('RGB'))

        boxes_coord_list = [
            upper_left_box_coordinates,
            upper_right_box_coordinates,
            bottom_left_box_coordinates,
            bottom_right_box_coordinates
        ]

        for idx, coords in enumerate(boxes_coord_list):
            box = img[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1], :]
            Image.fromarray(box).save(os.path.join(separated_out_dir,
                                                   os.path.split(frame)[-1].rsplit(".", maxsplit=1)[0] +
                                                   "_%d.png" % (idx + 1)))


root_dir = "/home/temp/schock/MGS/Videos"

base_out_dir = "/home/temp/schock/MGS/ExtractedFrames/Whole"


subdirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
subdirs.sort()
movie_files = []
for subdir in subdirs:
    movie_files += [os.path.join(subdir, x) for x in os.listdir(subdir) if (os.path.isfile(os.path.join(subdir, x)) and x.endswith(".MOV"))]
movie_files.sort()


movie_dict = {}

for mv_file in movie_files:
    movie_dict[os.path.split(mv_file)[-1].rsplit(".", maxsplit=1)[0]] = get_vid_points(mv_file, base_out_dir)

json.dump(movie_dict, os.path.join(base_out_dir, "movie_box_points.json"))

selection_file = ""

selected_movies = []
with open(selection_file) as f:
    content = f.readlines()


for line in content:
    selected_movies.append(line.strip("\n"))

selected_movies.sort()

with Pool() as p:
    p.map(partial(extract_boxes_from_frames, base_dir=base_out_dir), selected_movies)
