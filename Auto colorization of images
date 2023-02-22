import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imread,imshow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, MaxPool2D,UpSampling2D,Input, LeakyReLU
from tensorflow.keras.models import Sequential
import shutil #helps move files from one folder to another
import os #helps with the path

#training dataset

Train_root_dir='imagedata/trainingset/train_color' #dataset path
DEST='finalists/images' #create new folder for training imges

if not os.path.exists(DEST):
    os.makedirs(DEST)
    for img in os.listdir(os.path.join(Train_root_dir)):
        SOR= os.path.join(Train_root_dir,img)
        try:
            cv2.imread(SOR)
            shutil.copy(SOR,DEST)
        except:
            print(f"image at {SOR} cannot be used")

len (os.listdir("finalists/images"))

train_datagen= ImageDataGenerator(rescale=1/255,featurewise_center=True,rotation_range=40,horizontal_flip=True)
train=train_datagen.flow_from_directory("finalists",target_size=(256,256),batch_size=100,shuffle=True)

#to visualize the images

t_img,label= train.next()
def plotImage(img_arr,label):
    plt.figure(figsize=(5,5))
    for im, l in zip(img_arr,label):
        plt.imshow(im)
        plt.title(im.shape)
        plt.axis('off')
        plt.show()
        
plotImage(t_img[:10],label[:10])

t_img[:10].shape

X=[]
y=[]

#converting the images from rgb to lab and then appending them into two seperate numpy arrays


for img in t_img:
    try:
        lab =rgb2lab(img)
        X.append(lab[:,:,0])
        y.append(lab[:,:,1:]/128)
    except:
        print('error')
        
X_train= np.array(X)
X_train = np.expand_dims(X_train,axis=len(X_train.shape))
y_train=np.array(y)

X_train.shape,y_train.shape

#building encoder decoder models

#encoding
model = Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding ='same',strides=2,input_shape=(256,256,1)))
model.add(Conv2D(128,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding ='same',strides=2))
model.add(Conv2D(256,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(256,(3,3),activation='relu',padding ='same',strides=2))
model.add(Conv2D(512,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(512,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(256,(3,3),activation='relu',padding ='same'))

#decoding

model.add(Conv2D(128,(3,3),activation='relu',padding ='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding ='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(16,(3,3),activation='relu',padding ='same'))
model.add(Conv2D(2,(3,3),activation=LeakyReLU(alpha=0.01),padding ='same'))
model.add(UpSampling2D((2,2)))

model.summary()

model.compile(optimizer='adam',metrics=['acc'],loss='mse',run_eagerly=True)

his =model.fit(X_train,y_train,epochs=700,batch_size=32,steps_per_epoch=X_train.shape[0]//32,verbose=1)

model.save('./bestmodel1.h5')

#streamlit

%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb   #to convert the image into lab or rgb type
from skimage.io import imread,imshow   #to read images and display it
from keras.preprocessing.image import ImageDataGenerator #Generate batches of tensor image data with real-time data augmentation.
from keras.utils import img_to_array #converts a image into numpy array
from keras.utils import load_img,save_img #load the image frrom array
from keras.layers import Conv2D, MaxPool2D,UpSampling2D,Input, LeakyReLU #encoder layers
from keras.models import load_model
import shutil #helps move files from one folder to another
import os #helps with the path


st.title("Auto colorization of Images")
img1_color= []
path= st.file_uploader("Upload An Image", type=["png","jpg","jpeg"])
if st.button("upload"):
  st.text("Input Image")
  st.image(path)


  #loading the model
  
  model=load_model("model7.h5")
  img=img_to_array(
      load_img(path,target_size=(256,256,3)))/255
  plt.title("input image")
  imshow(img)
  plt.axis("off")
  plt.show()

  img1_color.append(img)
  
  #converting the image from rgb to lab
  
  img1_color= rgb2lab(img1_color)[:,:,:,0]
  img1_color = img1_color.reshape(img1_color.shape+(1,))
    
  #model prediction of the input
  
  output1 = model.predict(img1_color)
  output1 = output1*128

  result= np.zeros((256,256,3))
  result[:,:,0] = img1_color[0][:,:,0]
  result[:,:,1:] = output1[0]
  
  #converting the result from lab to rgb
  
  img1=lab2rgb(result)
  st.text("Output Image")
  plt.title("output image")
  imshow(img)
  st.image(img1)


! streamlit run app.py & npx localtunnel --port 8501
