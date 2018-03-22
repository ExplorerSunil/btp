import read_data
import random
np.random.seed(1337)
from training_model read_model
import sys
import scipy.io
import numpy as np


def predict():
   model = read_model()
   videos = []
   datapoints, words = make_test_samples()
   for datanumber in len(datapoints):
      videos.append(read_data.read_word(datapoint[datanumber], read_data.get_frame_length(words[datanumber])))
   print(model.predict(x=np.stack(videos), verbose=1, batch_size = len(datapoints)))



max = np.argmax(arr)




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
   return test_samples, [word[:-4] for word in rnd_words_names]


