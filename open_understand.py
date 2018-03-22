from os import listdir
import math
#import numpy as np
import string
#import scipy.io
#from keras.utils import np_utils, generic_utils
#import cv2
#from PIL import Image

speaker_numbers=[1,2,3,4,5,6,7,23,24]

for speaker_number in speaker_numbers:
    path = "data/alignments/s"+str(speaker_number)
    video_names = listdir(path)
    video_names.sort()
    for i in range(0, len(video_names)):
        videopath = path + "/" + video_names[i]
        with open(videopath, 'r') as f:
            lines = f.readlines()
        align = [(math.floor(int(y[0]) / 1000), math.ceil(int(y[1]) / 1000), y[2]) for y in [x.strip().split(" ") for x in lines]]
#        print(align)
        for k in align:
            st = str(speaker_number) + " "+str(video_names[i][0:6])+" " + str(k[0]) + ' ' + str(k[1])
            file_name = "words/" +  k[2] + ".txt"
            with open(file_name, "a") as myfile:
                myfile.write(st)
                myfile.write('\n')














