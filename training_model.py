from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import json
from keras.layers.wrappers import *
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Masking, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.optimizers import *
from keras.datasets import imdb
import time
from keras.models import model_from_json

NIL = 0.0


def save_model(model, save_weight_to, save_topo_to):
    json_string = model.to_json()
    model.save_weights(save_weight_to, overwrite=True)
    with open(save_topo_to, 'w') as outfile:
        json.dump(json.loads(json_string), outfile, indent=4)
        


## batchsize  x 40 x (60,80) ===> [40x(60x80)]

# Masking(mask_value=NIL)

def build_network(max_seqlen=None, image_size=(100, 50), fc_size=256,
                  save_weight_to='untrained_weight.h5', save_topo_to='untrained_topo.json', save_result=True,
                  lr=0.001, momentum=0.06,decay=0.0005,nesterov=True,
                  rho=0.9,epsilon=1e-6, 
                  optimizer='sgd', load_cache=False,   # the optimizer here could be 'sgd', 'adagrad', 'rmsprop'
                  cnn=False,dict_size=26,filter_length=5):

    try:
        if load_cache:
            return read_model(weights_filename=save_weight_to,
                              topo_filename=save_topo_to)
    except:
        pass
    
    
    start_time = time.time()    
    
    print("Creating Model...")    
    model = Sequential()

    if not cnn:        
        print("Adding TimeDistributeDense Layer...")    
        model.add(TimeDistributed(Dense(fc_size, activation='relu'), input_shape=(max_seqlen, image_size[0]*image_size[1])))
    else:
        print("Adding Convolution1D Layer...")
        model.add(Convolution1D(fc_size, filter_length,input_shape=(max_seqlen, image_size[0]*image_size[1])))

        # TODO
        # Reshape -> conv -> reshap
        model.add(TimeDistributed(Convolution1D(nb_filter, filter_length, activation='relu')))


    print("Adding Masking Layer...")
    model.add(Masking(mask_value=0.0))
    
    print("Adding First LSTM Layer...")
    model.add(LSTM(fc_size, return_sequences=True))

    print("Adding Second LSTM Layer...")
    model.add(LSTM(fc_size, return_sequences=False))

    print("Adding Final Dense Layer...")
    model.add(Dense(dict_size, activation='relu'))

    print("Adding Softmax Layer...")
    model.add(Activation('softmax'))

    print("Compiling the model to runnable code, which will take a long time...")
    if optimizer == 'sgd':
       optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=nesterov)
    if optimizer == 'rmsprop':
       optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon)
    if optimizer == 'adagrad':
       optimizer = Adagrad(lr=lr, epsilon=epsilon)
    
    ## Takes my macbook pro 1-2min to finish.    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    end_time = time.time()
    
    print("----- Compilation Takes %s Seconds -----" %  (end_time - start_time))


    
    if save_result:        
        print("Saving Model to file...")
        save_model(model, save_weight_to, save_topo_to)

    print("Finished!")
    return model





def train(model=None, 
          X_train=[], y_train=[],
          X_test=[], y_test=[], batch_size=32,
          iter_times=7, show_accuracy=True,
          save_weight_to='trained_weight.h5',
          save_topo_to='trained_topo.json',
          save_result=True, validation_split=0.1):
    
    if (not model) or len(X_train) == 0:
        print("Please provide legal input parameters!")
        return

    start_time = time.time()
    print("Training the model, which will take a long long time...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=iter_times, validation_split=validation_split)
    end_time = time.time()
    print("----- Training Takes %s Seconds -----" %  (end_time - start_time))


    print("Testing the model...")
#    print(X_test.shape, "and ", y_test.shape)
    score = model.evaluate(x=X_test, y=y_test)#batch_size=batch_size)

    print('Test score: {}'.format( score))
#    print('Test accuracy: not available')

    if save_result:        
        print("Saving Model to file...")
        save_model(model, save_weight_to, save_topo_to)

    print("Finished!")
    return score
    

def read_model(weights_filename='untrained_weight.h5',
               topo_filename='untrained_topo.json'):
    print("Reading Model from "+weights_filename + " and " + topo_filename)
    print("Please wait, it takes time.")
    with open(topo_filename) as data_file:
        topo = json.dumps(json.load(data_file))
        model = model_from_json(topo)
        model.load_weights(weights_filename)
        print("Finish Reading!")
        return model




#def test():
#    print (build_network(cnn=True, save_result=False))


## The data format we probably need:
### - Data:(totalDataNumber, maxSeqLen, 40x40)
### - Label:(totalDataNumber)
# test()
    
