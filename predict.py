from read_data import  get_frame_length, read_word
import random
import numpy as np
np.random.seed(1337)
from training_model import read_model 
import sys
from os import listdir



def predict():
   model = read_model()
   result = []
   datapoints, words = make_test_samples()
   for datanumber in range(len(datapoints)):
      video = np.stack(read_word(datapoints[datanumber], get_frame_length(words[datanumber])))
#      print(video.shape)
      result.append(model.predict(x=video.reshape(1, video.shape[0], video.shape[1]), verbose=0))
      print(result[-1][0,:].argmax() + "for word: " + words[datanumber])
#   for res in result:








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
         lines = file.readlines()
      datapoints = [(int(float(y[0])), y[1], int(float(y[2]))) for y in [x.strip().split(" ") for x in lines]]
      test_samples.append(random.choice(datapoints))
   return test_samples, [word[:-4] for word in rnd_words_names]



predict()
