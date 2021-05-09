#!/usr/bin/env python
# coding: utf-8

# In[5]:


from google.colab import files
uploaded = files.upload()


# In[6]:


get_ipython().run_line_magic('cd', '/content/drive/My Drive/Colab Notebooks/Modern AI Portfolio Builder/Emotion AI /')


# In[1]:




import pandas as pd
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow


# In[8]:



keyfacial_df = pd.read_csv('data.csv')


# In[7]:


keyfacial_df


# In[9]:



keyfacial_df.info()


# In[10]:



keyfacial_df.isnull().sum()


# In[11]:


keyfacial_df['Image'].shape


# In[12]:



keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x: np.fromstring(x, dtype = int, sep = ' ').reshape(96, 96))


# In[13]:



keyfacial_df['Image'][0].shape


# MINI CHALLENGE #1:
# - Obtain the average, minimum and maximum values for 'right_eye_center_x' 

# In[14]:


keyfacial_df.describe()


# # TASK #3: PERFORM IMAGE VISUALIZATION

# In[15]:



i = np.random.randint(1, len(keyfacial_df))
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')


# In[16]:



fig = plt.figure(figsize=(20, 20))

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)    
    image = plt.imshow(keyfacial_df['Image'][i],cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')
    


# MINI CHALLENGE #2: 
# - Perform a sanity check on the data by randomly visualizing 64 new images along with their cooresponding key points

# In[17]:


import random

fig = plt.figure(figsize=(20, 20))

for i in range(64):
    k = random.randint(1, len(keyfacial_df))
    ax = fig.add_subplot(8, 8, i + 1)    
    image = plt.imshow(keyfacial_df['Image'][k],cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[k][j-1], keyfacial_df.loc[k][j], 'rx')
    


# # TASK #4: PERFORM IMAGE AUGMENTATION

# In[18]:



import copy
keyfacial_df_copy = copy.copy(keyfacial_df)


# In[19]:




columns = keyfacial_df_copy.columns[:-1]
columns


# In[20]:



keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 1))


for i in range(len(columns)):
  if i%2 == 0:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x) )


# In[21]:



plt.imshow(keyfacial_df['Image'][0], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')


# In[22]:



plt.imshow(keyfacial_df_copy['Image'][0],cmap='gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# In[23]:



augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy))


# In[24]:


augmented_df.shape


# In[25]:




import random

keyfacial_df_copy = copy.copy(keyfacial_df)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x:np.clip(random.uniform(1.5, 2)* x, 0.0, 255.0))
augmented_df = np.concatenate((augmented_df, keyfacial_df_copy))
augmented_df.shape


# In[26]:




plt.imshow(keyfacial_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# MINI CHALLENGE #3:
# - Augment images by flipping them vertically 
# (Hint: Flip along x-axis and note that if we are flipping along x-axis, x co-ordinates won't change)

# In[27]:


keyfacial_df_copy = copy.copy(keyfacial_df)


# In[28]:


keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 0))

for i in range(len(columns)):
  if i%2 == 1:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x) )


# MINI CHALLENGE #4:
# - Perform a sanity check and visualize sample images

# In[29]:


plt.imshow(keyfacial_df_copy['Image'][0], cmap='gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# In[ ]:





# # TASK #5: PERFORM DATA NORMALIZATION AND TRAINING DATA PREPARATION

# In[30]:



img = augmented_df[:,30]


img = img/255.


X = np.empty((len(img), 96, 96, 1))

for i in range(len(img)):
  X[i,] = np.expand_dims(img[i], axis = 2)


X = np.asarray(X).astype(np.float32)
X.shape


# In[31]:



y = augmented_df[:,:30]
y = np.asarray(y).astype(np.float32)
y.shape


# In[32]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# MINI CHALLENGE #5: 
# - Try a different value for 'test_size' and verify that the split was successful

# In[33]:


X_train.shape


# In[34]:


X_test.shape


# # TASK #9: BUILD DEEP RESIDUAL NEURAL NETWORK KEY FACIAL POINTS DETECTION MODEL 

# In[35]:


def res_block(X, filter, stage):


  X_copy = X

  f1 , f2, f3 = filter

 
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)


  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  
  X_copy = X


  
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

 
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  
  X_copy = X


 
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X) 

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)


  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X


# In[36]:


input_shape = (96, 96, 1)


X_input = Input(input_shape)


X = ZeroPadding2D((3,3))(X_input)


X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)


X = res_block(X, filter= [64,64,256], stage= 2)


X = res_block(X, filter= [128,128,512], stage= 3)



X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)


X = Flatten()(X)
X = Dense(4096, activation = 'relu')(X)
X = Dropout(0.2)(X)
X = Dense(2048, activation = 'relu')(X)
X = Dropout(0.1)(X)
X = Dense(30, activation = 'relu')(X)


model_1_facialKeyPoints = Model( inputs= X_input, outputs = X)
model_1_facialKeyPoints.summary()


# In[ ]:





# In[3]:


from google.colab import files
uploaded = files.upload()


# In[40]:



checkpointer = ModelCheckpoint(filepath = "FacialKeyPoints_weights.hdf5", verbose = 1, save_best_only = True)


# In[ ]:





# In[42]:


history = model_1_facialKeyPoints.fit(X_train, y_train, batch_size = 32, epochs = 2, validation_split = 0.05, callbacks=[checkpointer])


# In[ ]:




model_json = model_1_facialKeyPoints.to_json()
with open("FacialKeyPoints-model.json","w") as json_file:
  json_file.write(model_json)


# In[ ]:





# # TASK #11: ASSESS TRAINED KEY FACIAL POINTS DETECTION MODEL PERFORMANCE

# In[ ]:


with open('detection.json', 'r') as json_file:
    json_savedModel= json_file.read()
    

model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
model_1_facialKeyPoints.load_weights('weights_keypoint.hdf5')
adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer= adam , metrics = ['accuracy'])


# In[ ]:




result = model_1_facialKeyPoints.evaluate(X_test, y_test)
print("Accuracy : {}".format(result[1]))


# In[ ]:



history.history.keys()


# In[ ]:


# Plot the training artifacts

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'], loc = 'upper right')
plt.show()


# # TASK #12: IMPORT & EXPLORE DATASET FOR FACIAL EXPRESSION DETECTION

# In[ ]:



facialexpression_df = pd.read_csv('icml_face_data.csv')


# In[ ]:


facialexpression_df


# In[ ]:


facialexpression_df[' pixels'][0] # String format


# In[ ]:




def string2array(x):
  return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')


# In[ ]:




def resize(x):
  
  img = x.reshape(48, 48)
  return cv2.resize(img, dsize=(96, 96), interpolation = cv2.INTER_CUBIC)


# In[ ]:


facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: string2array(x))


# In[ ]:


facialexpression_df[' pixels'] = facialexpression_df[' pixels'].apply(lambda x: resize(x))


# In[ ]:


facialexpression_df.head()


# In[ ]:



facialexpression_df.shape


# In[ ]:



facialexpression_df.isnull().sum()


# In[ ]:


label_to_text = {0:'anger', 1:'disgust', 2:'sad', 3:'happiness', 4: 'surprise'}


# In[ ]:


plt.imshow(facialexpression_df[' pixels'][0], cmap = 'gray')


# In[ ]:


emotions = [0, 1, 2, 3, 4]

for i in emotions:
  data = facialexpression_df[facialexpression_df['emotion'] == i][:1]
  img = data[' pixels'].item()
  img = img.reshape(96, 96)
  plt.figure()
  plt.title(label_to_text[i])
  plt.imshow(img, cmap = 'gray')


# In[ ]:


facialexpression_df.emotion.value_counts().index


# In[ ]:


facialexpression_df.emotion.value_counts()


# In[ ]:


plt.figure(figsize = (10,10))
sns.barplot(x = facialexpression_df.emotion.value_counts().index, y = facialexpression_df.emotion.value_counts())


# # TASK #14: PERFORM DATA PREPARATION AND IMAGE AUGMENTATION

# In[ ]:


X[0]


# In[ ]:


y


# In[ ]:



X = np.stack(X, axis = 0)
X = X.reshape(24568, 96, 96, 1)

print(X.shape, y.shape)


# In[ ]:


print(X_val.shape, y_val.shape)


# In[ ]:


print(X_Test.shape, y_Test.shape)


# In[ ]:


print(X_train.shape, y_train.shape)


# In[ ]:


X_train


# In[ ]:


train_datagen = ImageDataGenerator(
rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = "nearest")


# In[ ]:





# In[ ]:


input_shape = (96, 96, 1)


X_input = Input(input_shape)


X = ZeroPadding2D((3, 3))(X_input)


X = Conv2D(64, (7, 7), strides= (2, 2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides= (2, 2))(X)


X = res_block(X, filter= [64, 64, 256], stage= 2)


X = res_block(X, filter= [128, 128, 512], stage= 3)



X = AveragePooling2D((4, 4), name = 'Averagea_Pooling')(X)


X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)

model_2_emotion = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model_2_emotion.summary()


# In[ ]:



model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[ ]:



earlystopping = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 20)


checkpointer = ModelCheckpoint(filepath = "FacialExpression_weights.hdf5", verbose = 1, save_best_only=True)


# In[ ]:


history = model_2_emotion.fit(train_datagen.flow(X_train, y_train, batch_size=64),
	validation_data=(X_val, y_val), steps_per_epoch=len(X_train) // 64,
	epochs= 2, callbacks=[checkpointer, earlystopping])


# In[ ]:




model_json = model_2_emotion.to_json()
with open("FacialExpression-model.json","w") as json_file:
  json_file.write(model_json)


# In[ ]:


with open('emotion.json', 'r') as json_file:
    json_savedModel= json_file.read()
    

model_2_emotion = tf.keras.models.model_from_json(json_savedModel)
model_2_emotion.load_weights('weights_emotions.hdf5')
model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


# In[ ]:


score = model_2_emotion.evaluate(X_Test, y_Test)
print('Test Accuracy: {}'.format(score[1]))


# In[ ]:


history.history.keys()


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[ ]:


epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


# In[ ]:


plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()


# In[ ]:



predicted_classes = np.argmax(model_2_emotion.predict(X_Test), axis=-1)
y_true = np.argmax(y_Test, axis=-1)


# In[ ]:


y_true.shape


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, predicted_classes)
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, cbar = False)


# In[ ]:


L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i].reshape(96,96), cmap = 'gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)   


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_true, predicted_classes))


# # PART 3. COMBINE BOTH FACIAL EXPRESSION AND KEY POINTS DETECTION MODELS

# # TASK #18: COMBINE BOTH MODELS (1) FACIAL KEY POINTS DETECTION AND (2) FACIAL EXPRESSION MODELS

# In[ ]:


def predict(X_test):

 
  df_predict = model_1_facialKeyPoints.predict(X_test)


  df_emotion = np.argmax(model_2_emotion.predict(X_test), axis=-1)


  df_emotion = np.expand_dims(df_emotion, axis = 1)

  
  df_predict = pd.DataFrame(df_predict, columns= columns)

  
  df_predict['emotion'] = df_emotion

  return df_predict


# In[ ]:


df_predict = predict(X_test)


# In[ ]:


df_predict.head()


# MINI CHALLENGE #17: 
# - Plot a grid of 16 images along with their predicted emotion and facial key points

# In[ ]:




fig, axes = plt.subplots(4, 4, figsize = (24, 24))
axes = axes.ravel()

for i in range(16):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediction = {}'.format(label_to_text[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')
            


# In[ ]:


import json
import tensorflow.keras.backend as K

def deploy(directory, model):
  MODEL_DIR = directory
  version = 1 


  export_path = os.path.join(MODEL_DIR, str(version))
  print('export_path = {}\n'.format(export_path))


  if os.path.isdir(export_path):
    print('\nAlready saved a model, cleaning up\n')
    get_ipython().system('rm -r {export_path}')

  tf.saved_model.save(model, export_path)

  os.environ["MODEL_DIR"] = MODEL_DIR


# In[ ]:



get_ipython().system('echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -')
get_ipython().system('apt update')


# In[ ]:



get_ipython().system('apt-get install tensorflow-model-server')


# In[ ]:


deploy('/model', model_1_facialKeyPoints)


# In[ ]:


get_ipython().run_cell_magic('bash', '--bg ', 'nohup tensorflow_model_server \\\n  --rest_api_port=4500 \\\n  --model_name=keypoint_model \\\n  --model_base_path="${MODEL_DIR}" >server.log 2>&1')


# In[ ]:


get_ipython().system('tail server.log')


# In[ ]:


deploy('/model1', model_2_emotion)


# In[ ]:


get_ipython().run_cell_magic('bash', '--bg ', 'nohup tensorflow_model_server \\\n  --rest_api_port=4000 \\\n  --model_name=emotion_model \\\n  --model_base_path="${MODEL_DIR}" >server.log 2>&1')


# In[ ]:


get_ipython().system('tail server.log')


# - **Congratulations! now we have successfully loaded a servable version of our model {name: keypoint_model version: 1}** 
# - **Congratulations! now we have successfully loaded a servable version of our model {name: emotion_model version: 1}** 

# # TASK #21: MAKE REQUESTS TO MODEL IN TENSORFLOW SERVING

# In[ ]:


import json


data = json.dumps({"signature_name": "serving_default", "instances": X_test[0:3].tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


# In[ ]:


get_ipython().system('pip install -q requests')


# In[ ]:


import requests


def response(data):
  headers = {"content-type": "application/json"}
  json_response = requests.post('http://localhost:4500/v1/models/keypoint_model/versions/1:predict', data=data, headers=headers, verify = False)
  df_predict = json.loads(json_response.text)['predictions']
  json_response = requests.post('http://localhost:4000/v1/models/emotion_model/versions/1:predict', data=data, headers=headers, verify = False)
  df_emotion = np.argmax(json.loads(json_response.text)['predictions'], axis = 1)
  

  df_emotion = np.expand_dims(df_emotion, axis = 1)

 
  df_predict= pd.DataFrame(df_predict, columns = columns)


  df_predict['emotion'] = df_emotion

  return df_predict


# In[ ]:



df_predict = response(data)


# In[ ]:


df_predict


# In[ ]:




fig, axes = plt.subplots(3, 1, figsize = (24, 24))
axes = axes.ravel()

for i in range(3):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediction = {}'.format(label_to_text[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')
            


# # EXCELLENT JOB! NOW YOU HAVE A SOLID KNOWLEDGE OF EMOTION AI! YOU SHOULD BE SUPER PROUD OF YOUR NEWLY ACQUIRED SKILLS :)

# # MINI CHALLENGE SOLUTIONS

# MINI CHALLENGE #1:
# - Obtain the average, minimum and maximum values for 'right_eye_center_x' 

# In[ ]:


keyfacial_df.describe()


# MINI CHALLENGE #2: 
# - Perform a sanity check on on the data by randomly visualizing 64 new images along with their cooresponding key points

# In[ ]:


import random

fig = plt.figure(figsize=(20, 20))

for i in range(64):
    k = random.randint(1, len(keyfacial_df))
    ax = fig.add_subplot(8, 8, i + 1)
    image = plt.imshow(keyfacial_df['Image'][k],cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[k][j-1], keyfacial_df.loc[k][j], 'rx')
    


# MINI CHALLENGE #3:
# - Augment images by flipping them vertically 
# (Hint: Flip along x-axis and note that if we are flipping along x-axis, x co-ordinates won't change)

# In[ ]:


keyfacial_df_copy = copy.copy(keyfacial_df)


keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 0))


for i in range(len(columns)):
  if i%2 == 1:
    keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x) )


# In[ ]:



plt.imshow(keyfacial_df_copy['Image'][0],cmap='gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')


# In[ ]:


print('Train size =', X_train.shape)
print('Test size =', X_test.shape)


# In[ ]:


plt.imshow(facialexpression_df[' pixels'][0], cmap = 'gray')


# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(x = facialexpression_df.emotion.value_counts().index, y = facialexpression_df.emotion.value_counts() )
plt.title('Number of images per emotion')


# In[ ]:


train_datagen = ImageDataGenerator(
rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True,
    brightness_range = [1.1, 1.5],
    fill_mode = "nearest")

# Note on "Brightness_range"
# 1.0 does not affect image brightness
# numbers less than 1.0 darken the image [0.5, 1.0]
# numbers larger than 1.0 brighten the image [1.0, 1.5] 


# In[ ]:


L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (24, 24))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i].reshape(96,96), cmap = 'gray')
    axes[i].set_title('Prediction = {}\n True = {}'.format(label_to_text[predicted_classes[i]], label_to_text[y_true[i]]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)   


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_true, predicted_classes))


# In[ ]:


# Plotting the test images and their predicted keypoints and emotions

fig, axes = plt.subplots(4, 4, figsize = (24, 24))
axes = axes.ravel()

for i in range(16):

    axes[i].imshow(X_test[i].squeeze(),cmap='gray')
    axes[i].set_title('Prediction = {}'.format(label_to_text[df_predict['emotion'][i]]))
    axes[i].axis('off')
    for j in range(1,31,2):
            axes[i].plot(df_predict.loc[i][j-1], df_predict.loc[i][j], 'rx')
            

