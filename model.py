"Import required libraries"
#natural langauge processing
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')
#tensorflow libs
import tflearn
import tensorflow as tf
#pickle to save data
import pickle
#import time and path
import os.path, time
#other libs
import numpy as np 
import json
import random

class Model:

    '''
    The class contains methods to check if the intents.json
    file is modified. If the file is modified then the model
    will be retrained else previous model will be returned.
    '''

    def __check_file():
        '''
        Function to check if the intents.json file was modified. If
        the file is modified the function also logs the modification time
        into the logs

        INPUT:
            1. NONE

        OUTPUT:
            1. value (string): 'modified' if the true 'unmodified' if false
        '''
        #get last modification time from logs
        with open("logs/logs.txt", "r") as logs:
            log_time = logs.read()
        #get the last modification time from intents file
        intents_time = time.ctime(os.path.getmtime("data/intents.json"))

        #check if the file was modified
        if log_time == intents_time:
            return "unmodified"
        else:
            #write the modified time to logs
            with open("logs/logs.txt", "w") as text_file:
                text_file.write(time.ctime(os.path.getmtime("data/intents.json")))
            return "modified"

    def __dnn_files():
        '''
        Function to read the files required for DNN model.

        INPUT:
            NONE

        OUTPUT:
            1. words, labels, training, output: required files for dnn
        '''
        #load the pickled file
        with open("data/model_data.pkl", "rb") as f:
            words, labels, training, output = pickle.load(f)

        return words, labels, training, output

    def get_model():
        '''
        Function to train and load the model if the file is modified else 
        just load the model and return it.

        INPUT:
            1. NONE

        OUTPUT:
            1. model(tflearn-model): DNN model
        '''
        #call the method to check if the file was modified
        mod_val = Model.__check_file()

        if mod_val == "modified":
            #initialize stemmer
            stemmer = LancasterStemmer()
            #load json file
            with open("data/intents.json") as file:
                data = json.load(file)
            
            #lists to store values
            words = []
            labels = []
            docs_x = []
            docs_y = []

            #stemming
            for intent in data['intents']:
                for pattern in intent['patterns']:
                    #tokenize
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent['tag'])

                if intent['tag'] not in labels:
                    labels.append(intent['tag'])
            
            #save the stemmed words and associated labels
            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))
            labels = sorted(labels)

            #create bag of words
            training = []
            output = []
            #list with initial 0s
            out_empty = [0 for _ in range(len(labels))]
            #loop through doc_X
            for x, doc in enumerate(docs_x):
                bag = []
                
                wrds = [stemmer.stem(w) for w in doc]

                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                out_row = out_empty[:]
                out_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(out_row)

            training = np.array(training)
            output = np.array(output)

            #save the files
            with open("data/model_data.pkl", "wb") as f:
                pickle.dump((words, labels, training, output), f)

        #train the model
        #get the dnn files
        words, labels, training, output = Model.__dnn_files()
        #clear the default graph stack and resets the global default graph.
        tf.compat.v1.reset_default_graph()
        #layers
        net = tflearn.input_data(shape = [None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
        net = tflearn.regression(net)
        #DNN Model
        model = tflearn.models.dnn.DNN(net)

        #train and save the model if the file is modified
        if mod_val == "modified":
            model.fit(training, output, batch_size = 8, n_epoch = 1000, show_metric = True)
            #save the model
            model.save("model/model.tflearn")

        #load the model
        model.load("model/model.tflearn")

        return model, words, labels

    def bag_of_words(s, words, stemmer):
        '''
        Function to generate bag of words
        ''' 
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(w.lower()) for w in s_words if w]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
        
        return np.array(bag)

    def get_reply(msg):
        '''
        Function to make predictions and generate a reply

        INPUT:
            1. msg('String'): message from the user
        
        OUTPUT:
            2. reply('String'): predicted reply to the message from user
        '''
        #get the model and other variables
        model, words, labels = Model.get_model()
        stemmer = LancasterStemmer()
        #load json file
        with open("data/intents.json") as file:
            data = json.load(file)

        #get the bag of words
        results = model.predict([Model.bag_of_words(msg, words, stemmer)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        #threshold
        if results[results_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            
            return random.choice(responses)
        else:
            return "Unable to answer! Please try different question!"

