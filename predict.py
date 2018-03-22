import read_data
import random
np.random.seed(1337)
import training_model
import sys
import scipy.io
import numpy as np

model = training_model.read_model()

def predict(filepath, verbose=1):
   f = scipy.io.loadmat(filepath)
   video_frames = int(f.get('siz')[0,2])
   video = f.get('vid').flatten(order='f').reshape((video_frames, 4800))
   for frame in range(video_frames):
      video[frame, :] = video[frame, :].reshape(100, 50).flatten(order='f')
#               cv2.imshow("datafile", video[frame, :].reshape(60, 80))
#               cv2.waitKey(25)
#appending video with 0's if shorter than max_seqlen frames
   max_seqlen = 75
   if video_frames < max_seqlen:
      append = np.zeros((max_seqlen-video_frames, 4800))
      video = np.concatenate((video, append))
   global model
   return model.predict(x=video.reshape((1, video.shape[0], video.shape[1])), verbose=verbose)

arr = predict(sys.argv[1])
print(arr)
max = np.argmax(arr)
#for i in range(0, len(arr)):
#    if max <= arr[i]:
print("the spoken alphabet is {0} with probability {1}".format(chr(65+max), arr[0, max]))

def make_test_samples():
   path = "words"
   words_names = listdir(path)
   rnd_words_names = []
   for i in range(0,5):
      rnd_words_names.append(random.choice(words_names))
   print(rnd_words_names)
   test_samples=[]
   for i in rnd_words_names:
      with open("words/" + i, 'r') as file:
         lines = file.readline()
      datapoints = [(int(float(y[0])), y[1], int(float(y[2]))) for y in [x.strip().split(" ") for x in lines]]
      test_samples.append(random.choice(datapoints))
   return test_samples,rnd_words_names

def make_test_database(test_samples , rnd_words_names):
