import cv2
import os
import sys
from os.path import isfile, join

def convert_2_mp4(img_dir, mp4_path):
    frame_array = []
    files = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f))]  # for sorting the file names properly
    files.sort(key=lambda x: int(x[:-4]))
    for i in range(len(files)):
        filename = img_dir + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        fps = 30

        # inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    convert_2_mp4(sys.argv[1], sys.argv[2])