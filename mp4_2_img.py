import cv2
import sys


def convert_2_frame(mp4_path, img_dir):
    vidcap = cv2.VideoCapture(mp4_path)
    success, image = vidcap.read()
    count = 0
    while success:
        str_c = str(count)
        cv2.imwrite(img_dir+'frame'+(5-len(str_c))*'0'+str_c+'.jpg', image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1

if __name__ == '__main__':
    convert_2_frame(sys.argv[1], sys.argv[2])