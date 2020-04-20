import csv
import os
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt

from general import GenerateOptions
from general import utils
from generate import generate


fg_img_path_base = "C:/workspace/consolsys/InnoLab/Mykad/cropped"
bg_img_path_base = "I:/train/bg"
out_img_path_base = "I:/train/generated"
labels_file_path = "I:/train/labels.csv"


fg_img_paths = utils.get_all_file_paths(fg_img_path_base, ".jpg")
bg_img_paths = utils.get_all_file_paths(bg_img_path_base, ".jpg")


def get_random_fg_img():
    i = random.randint(0, len(fg_img_paths) - 1)
    fg_img_path = fg_img_paths[i]
    fg_img= utils.read_image(fg_img_path)
    return fg_img


def get_random_bg_img():
    i = random.randint(0, len(bg_img_paths) - 1)
    bg_img_path = bg_img_paths[i]
    bg_img = utils.read_image(bg_img_path)
    bg_img = utils.rescale(bg_img, scale=3.0)
    return bg_img


def from_row(row):
    img_filename = row[0]
    points = [(float(row[i * 2 + 1]), float(row[i * 2 + 2])) for i in range(4)]
    return img_filename, points


def to_row(img_filename, points):
    row = [img_filename]
    for point in points:
        row.append(str(point[0]))
        row.append(str(point[1]))
    return row


def preview_generated_files():
    with open(labels_file_path, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            filename, points = from_row(row)
            img_filepath = os.path.join(out_img_path_base, filename)
            img = cv2.imread(img_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fig, ax = plt.subplots()
            ax.fill([p[0] for p in points], [p[1] for p in points], edgecolor="b", fill=False)
            ax.imshow(img)
            plt.show()



if __name__ == "__main__":
    options = GenerateOptions()
    options.glare_img_path = './glare_images'
    options.random_distortion = False

    random.seed(137)

    # with open(labels_file_path, mode="w") as csv_file:
    #     csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     field_names = ["filename"]
    #     for i in range(4):
    #         field_names.append(f"point{i}")
    #     csv_writer.writerow(field_names)

    #     for i in range(8000):
    #         fg_img = get_random_fg_img()
    #         bg_img = get_random_bg_img()

    #         new_img, points = generate(bg_img, fg_img, options)

    #         new_img_filename = f"{i:06d}.jpg"

    #         new_img_filepath = os.path.join(out_img_path_base, new_img_filename)
    #         cv2.imwrite(new_img_filepath, new_img)
            
    #         csv_writer.writerow([new_img_filename, *points])

    preview_generated_files()

    # while True:
    #     fg_img = get_random_fg_img()
    #     bg_img = get_random_bg_img()

    #     new_img, points = generate(bg_img, fg_img, options)

    #     # for point in points:
    #     #     cv2.circle(new_img, tuple(point), 3, (0, 0, 255), -1)
        
    #     cv2.imshow("New", utils.rescale(new_img, .5))
        
    #     k = cv2.waitKey(0)
    #     if k == 27:
    #         break

    # cv2.destroyAllWindows()
    