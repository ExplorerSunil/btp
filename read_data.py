from os import listdir
import numpy as np
import sys
import scipy
import math
import scipy.io
from keras.utils import np_utils, generic_utils
import cv2
import training_model


file_names = listdir("words/")
file_names.sort()



labels =[i for i in range(0,53)]
#print(labels)
label_size = 53
#net = training_model.build_network(dict_size=label_size)
#speakers = 7

def read_word(datapoint, frame_length):
   video = []
   skip = False
   path = "target/data/s" + str(datapoint[0]) + "/" + datapoint[1]
   for num in range(datapoint[2], min(datapoint[2] + frame_length, 75)):
      filepath = path + "/mouth_" +  str(num).zfill(3) + ".png"
      image = cv2.imread(filepath)
#         print(image.shape, file = sys.stderr)
      if image is None:
         skip = True
         print("error in processing image: ", filepath, file=sys.stderr)
         break
      image = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
      if image.shape ==  (50,100):
         image = np.reshape(image,5000)
      video.append(image)

   if skip == True:
      video.clear()
      return video

#extending frames to frame_length
   frames_left = frame_length-len(video)
   for i in range(0, frames_left):
      video.append(video[len(video)-1])

   return video




def read_word_database(word,data_num):
   frame_length = get_frame_length(word)
   with open("words/" + word + ".txt", 'r') as file:
      lines = file.readlines()

   skip = False
#   print(str(len(lines)) + "data points found for word: " + word)
#check whether last line is empty
   datapoints = [(int(float(y[0])), y[1], int(float(y[2]))) for y in [x.strip().split(" ") for x in lines]]

   videos = []
   if data_num>0 :
      datapoints = datapoints[0:data_num]
   for datapoint in datapoints:
      video = read_word(datapoint, frame_length)
      video_array = np.stack(video)
      if video_array.shape != (frame_length, 5000):
         print("invalid video encountered: ", datapoint[1], "with shape: ", video_array.shape)
      else:
         videos.append(np.stack(video))

      video.clear()

   return np.stack(videos), np.tile(Label()[file_names.index(word + ".txt"), :], (len(videos), 1))

#print(read_word_database('please', 13).shape)

"""
def Data(speaker_number,max_seqlen=75, image_size=5000, data_num=1000):
   print("reading {0} video from speaker {1} from database".format(data_num,speaker_number))
   path = "target/data/s" + str(speaker_number)
   video_names = listdir(path)
   if len(video_names) != 1000:
      print("ERROR!!!: not enough videos... number of videos : {} ".format(len(video_names)))

   video_names.sort()

   video = []
   videos = []

   for i in range(0, len(video_names)):
      videopath = path + "/" + video_names[i]
      frame_names = listdir(videopath)
      frame_names.sort()
      temp = 0
      if  len(frame_names) > 75:
         temp = len(frame_names) - 75
      for j in range(len(frame_names)-temp):
         imagepath = videopath + "/" + frame_names[j]
#         print(imagepath)
         img = cv2.imread(imagepath)
#         print(img.shape)
         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         video.append(gray_img)
         imagepath = None

#extending frames if not equals to 75

      for j in range(0, 75-len(frame_names)):
         video.append(video[len(frame_names) +j-1])

#stacking all frames into array and appending 1000-videos list
      video_array = np.stack(video)
      if video_array.shape !=   (75,50,100):
         print("invalid video ", video_names[i], "with shape", video_array.shape)
      else:
         videos.append(video_array)

#      print(videos[len(videos)-1].shape, video_names[i])
#needs to be cleared so that new video can be stored (otherwise it would append to previous one)
      video.clear()
      videopath = None    #needs to be cleared for same reason
   return np.stack(videos)

"""

def get_frame_length(word):
   with open("find_mode.txt", 'r') as file:
      lines = file.readlines()
   #print(str(len(lines)) + "data points found for word: " + word)
   # check whether last line is empty
   word = word + ".txt"
   datapoints = [(y[0],int(y[1])) for y in [x.strip().split(" ") for x in lines]]
   for datapoint in datapoints:
      if datapoint[0] ==  word :
         return datapoint[1]

   print("word is not present in find_mode.txt file")






def Label(data_num=1000,label_size=53):
#   print(len(labels))
   return np_utils.to_categorical(np.array(labels))


def Train(word,data_num=-1,label_size=53):
#   print("training")
    #return Data(speaker_number,data_num=data_num), Label(data_num=data_num,label_size=label_size)
   x = read_word_database(word,data_num=data_num)
   print(x[1].shape, "and elements are:")
   print(x[1])
   return x


def Test(word,data_num=25,label_size=53):
#   print("testing")
    #return Data(speaker_number,data_num=data_num), Label(data_num=data_num,label_size=label_size)
   return read_word_database(word, data_num=data_num)

def Val(word,data_num=25,label_size=53):
   #return Data(data_num=data_num), Label(data_num=data_num,label_size=label_size)
   return read_word_database(word,data_num=data_num), Label(data_num=data_num, label_size=label_size)








net = training_model.build_network(dict_size = label_size)

for i in range(len(file_names)):
   X_train, y_train = Train(file_names[i][:-4],data_num=-1)
   X_test, y_test = Test(file_names[i][:-4], data_num=25)
   training_model.train(model=net,
                           X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test, iter_times=20)
   print("training for " + str(i) + "finished: " + file_names[i])
