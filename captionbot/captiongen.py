# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:16:48 2020

@author: ABHINAV
"""

import os
import string
import glob
import tensorflow.keras.applications.mobilenet  
import tqdm as tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
import tensorflow.keras.preprocessing.image
import pickle
from time import time
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras import Input

from tensorflow.keras.models import Model,load_model,save_model

from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


####################################################################################
'''
                                PREPROCESSING
'''
####################################################################################
START = "startseq"
STOP = "endseq"
EPOCHS = 10
root_captioning = "captions"

null_punct = str.maketrans('', '', string.punctuation)
lookup = dict()

with open( os.path.join(root_captioning,'Flickr8k_text','Flickr8k.token.txt'), 'r') as fp:
  
  max_length = 0
  for line in fp.read().split('\n'):
    tok = line.split()
    if len(line) >= 2:
      id = tok[0].split('.')[0]
      desc = tok[1:]
      
      # Cleanup description
      desc = [word.lower() for word in desc]
      desc = [w.translate(null_punct) for w in desc]
      desc = [word for word in desc if len(word)>1]
      desc = [word for word in desc if word.isalpha()]
      max_length = max(max_length,len(desc))
      
      if id not in lookup:
        lookup[id] = list()
      lookup[id].append(' '.join(desc))
      
lex = set()
for key in lookup:
  [lex.update(d.split()) for d in lookup[key]]

#################### PRINTING
'''
print(len(lookup)) # How many unique words
print(len(lex)) # The dictionary
print(max_length) # Maximum length of a caption (in words)
'''

img = glob.glob(os.path.join(root_captioning,'Flicker8k_Dataset', '*.jpg'))
len(img)


train_images_path = os.path.join(root_captioning,'Flickr8k_text','Flickr_8k.trainImages.txt') 
train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
test_images_path = os.path.join(root_captioning,'Flickr8k_text','Flickr_8k.testImages.txt') 
test_images = set(open(test_images_path, 'r').read().strip().split('\n'))

train_img = []
test_img = []


# building a path for test and train images
for i in img:
  f = os.path.split(i)[-1]
  if f in train_images: 
    train_img.append(f) 
  elif f in test_images:
    test_img.append(f)

#################### PRINTING
'''
print(len(train_images))
print(len(test_images))
'''

train_descriptions = {k:v for k,v in lookup.items() if f'{k}.jpg' in train_images}
for n,v in train_descriptions.items(): 
  for d in range(len(v)):
    v[d] = f'{START} {v[d]} {STOP}'


#len(train_descriptions)


encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048

#The preprocess_input function is meant to adequate your image to the format the model requires.
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

######## SUMMARY OF INCEPTION
#encode_model.summary()


def encodeImage(img):
  
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS) # Resize all images to a standard size   
  x = tensorflow.keras.preprocessing.image.img_to_array(img) # Convert a PIL image to a numpy array  
  x = np.expand_dims(x, axis=0) # Expand to 2D array  
  x = preprocess_input(x) # preprocessing needed by InceptionV3 or others
  # Calling InceptionV3 to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image  
  x = np.reshape(x, OUTPUT_DIM )
  return x

################ ENCODING IMAGE IF PICKLE IS NOT AVAILABLE
'''
encoding_train = {}
for id in tqdm(train_img):
   image_path = os.path.join(root_captioning,'Flicker8k_Dataset', id)
   img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
   encoding_train[id] = encodeImage(img)
   with open('train', "wb") as fp:
       pickle.dump(encoding_train, fp)
'''
with open('train', "rb") as fp:
   encoding_train = pickle.load(fp)

'''
encoding_test = {}
for id in tqdm(test_img):
   image_path = os.path.join(root_captioning,'Flicker8k_Dataset', id)
   img = tensorflow.keras.preprocessing.image.load_img(image_path, target_size=(HEIGHT, WIDTH))
   encoding_test[id] = encodeImage(img)
   with open('test', "wb") as fp:
    pickle.dump(encoding_test, fp)
'''
with open('test', "rb") as fp:
   encoding_test = pickle.load(fp)
    
# all training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)


word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
#print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1
    
vocab_size = len(idxtoword) + 1 
vocab_size

max_length +=2
#print(max_length)


####################################################################################
'''
                            TRAINING MODEL
'''
####################################################################################

# x1 - Training data for photos
# x2 - The caption that goes with each photo
# y - The predicted rest of the caption

def data_generator(descriptions, photos, wordtoidx, max_length, num_photos_per_batch):
  x1, x2, y = [], [], []
  n=0
  while True:
    for key, desc_list in descriptions.items():
      n+=1
      photo = photos[key+'.jpg']
      # Each photo has 5 descriptions
      for desc in desc_list:
        # Convert each word into a list of sequences.
        seq = [wordtoidx[word] for word in desc.split(' ') if word in wordtoidx]
        # Generate a training case for every possible sequence and outcome
        for i in range(1, len(seq)):
          in_seq, out_seq = seq[:i], seq[i]
          in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
          out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
          x1.append(photo)
          x2.append(in_seq)
          y.append(out_seq)
      if n==num_photos_per_batch:
        yield ([np.array(x1), np.array(x2)], np.array(y))
        x1, x2, y = [], [], []
        n=0

embedding_dim = 200
inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
embedding_dim

#caption_model.summary()

caption_model.compile(loss='categorical_crossentropy', optimizer='adam')


number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath

'''
for i in range(EPOCHS):
      generator = data_generator(train_descriptions, encoding_train, wordtoidx, max_length, number_pics_per_bath)
      caption_model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)  
caption_model.save_weights('models.hdf5')
'''


####################################################################################
'''
                                    PREDICTION
'''
####################################################################################
#caption_model.save('saved_model')
caption_model.load_weights('models.hdf5')
#caption_model=load('saved_model')
def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
'''
# for test images
import gtts
from playsound import playsound
for z in range(1,10):
  pic = list(encoding_test.keys())[z]
  image = encoding_test[pic].reshape((1,OUTPUT_DIM))
  print(os.path.join(root_captioning,'Flicker8k_Dataset', pic))
  x=plt.imread(os.path.join(root_captioning,'Flicker8k_Dataset', pic))
  plt.imshow(x)
  plt.show()
  capti=generateCaption(image)
  print("Caption:",capti)
  print("_____________________________________")
  tts = gtts.gTTS(str(capti), lang="en")
  tts.save("h{}.mp3".format(z))
  playsound("h{}.mp3".format(z))
'''
'''
# for individual image
img = tensorflow.keras.preprocessing.image.load_img('img.jpg', target_size=(HEIGHT, WIDTH))
img=encodeImage(img)
img=img.reshape((1,OUTPUT_DIM))
capti=generateCaption(img)
print(capti)
'''























