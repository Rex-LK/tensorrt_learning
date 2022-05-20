import cv2
import os
from PIL import Image


def PicToVideo(imgPath, videoPath):
    images = os.listdir(imgPath)
    print(len(images))
    fps = 10  # 帧率
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    im = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, im.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])
        for i in range(90):
            videoWriter.write(frame)
    videoWriter.release()


if __name__ == "__main__":

    imgPath = "/home/rex/Pictures/Wallpapers/"
    videoPath = "/home/rex/Desktop/test.avi"
    PicToVideo(imgPath, videoPath)
