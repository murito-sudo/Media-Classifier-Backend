#!/usr/bin/env python
# coding: utf-8

# In[112]:


from flask import Flask
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
import requests
from io import BytesIO
plt.style.use('classic')
#############################################################
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
#from keras import backend as K
####################################################
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd

import requests,random,json,time,webbrowser


# In[42]:



SIZE = 32
dataset = []   
label = [] 


# In[3]:

if not os.path.exists('NSFW.h5'):
    NSFW_images = os.listdir(os.getcwd() + '/NSFW/')
   
    for i, image_name in enumerate(NSFW_images):    # Iterate through the NSFW Folder
      
        try:
            if (image_name.split('.')[1] == 'png' or image_name.split('.')[1] == 'jpg'): #only add png and jpg to the dataset
                image = cv2.imread(os.getcwd() + '/NSFW/' + image_name) 
                image = Image.fromarray(image, 'RGB') #add the image as an array with format RGB
                image = image.resize((SIZE, SIZE))  #32x32 image resize to avoid bias
                dataset.append(np.array(image))  # append the numpy array image to the dataset
                label.append(1)  ##appending label
               
        except:
            pass
            
    
    
    # In[4]:
    
    
    SFW_images = os.listdir(os.getcwd() + '/SFW/')
    for i, image_name in enumerate(SFW_images):    #Iterate through the SFW Folder
        try:
            if (image_name.split('.')[1] == 'png' or image_name.split('.')[1] == 'jpg'): #only add jpg and png to the dataset
                image = cv2.imread(os.getcwd() + '/SFW/' + image_name) #read image
                image = Image.fromarray(image, 'RGB')  #add the image as an array with format RGB
                image = image.resize((SIZE, SIZE))  #32x32 image resize to avoid bias
                dataset.append(np.array(image))  #append the numpy array image to the dataset
                label.append(0)  # appending label label
        except:
            pass
        


    dataset = np.array(dataset)
    label = np.array(label)
    # convert array into dataframe
    
    
    
    # In[6]:
    
    
    from sklearn.model_selection import train_test_split
    #from keras.utils import to_categorical
    
    X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 25, train_size = 0.8)


    from keras.utils import normalize
    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)
    
    


# In[8]:


INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)
    
    
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

    
model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

    
model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

    
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
    
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[9]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',             #also try adam
              metrics=['accuracy'])

print(model.summary())    


# In[128]:


if os.path.exists('NSFW.h5'):
    model = load_model('NSFW.h5')
else:
    history = model.fit(X_train, 
                             y_train, 
                             batch_size = 64, 
                             verbose = 1, 
                             epochs = 20,      
                             validation_data=(X_test,y_test),
                             shuffle = False
                         )
    model.save('NSFW.h5')
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# In[130]:

    
    
    n=4#Select the index of image to be loaded for testin
    img = X_test[n]
    plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    print("The prediction for this image is: ", model.predict(input_img).astype(float))
    print("The actual label for this image is: ", y_test[n])
    
    
    
    # In[131]:
    
    
    #cuttle-environment-disable Classifier-api
    from keras.models import load_model
    # load model
    model = load_model('NSFW.h5')
    
    
    
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")
    
    


    # In[13]:
    
    
    mythreshold=0.908
    from sklearn.metrics import confusion_matrix
    
    y_pred = (model.predict(X_test)>= mythreshold).astype(int)
    cm=confusion_matrix(y_test, y_pred)  
    print(cm)
    
   


# In[14]:


    from sklearn.metrics import roc_curve
    y_preds = model.predict(X_test).ravel()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_preds)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'y--')
    plt.plot(fpr, tpr, marker='.')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.show()


# In[15]:

    import pandas as pd
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
    ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
    print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 


# In[16]:


    from sklearn.metrics import auc
    auc_value = auc(fpr, tpr)
    print("Area under curve, AUC = ", auc_value)


# In[132]:


def analyze_image(link):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    img = img.convert("RGB")
    img = img.resize((SIZE, SIZE))
    numpydata = np.array(img)
    input_img = np.expand_dims(numpydata, axis=0)
    print("The prediction for this image is: ", model.predict(input_img))
    plt.imshow(img)

    if model.predict(input_img)  == 0:
        print("image is probably safe for all audience")
    else:
        print("image is probably unsafe to certain audiences. NSFW(+18)")
    
    return [link, model.predict(input_img).tolist()]


# In[143]:


boards = ['a','c','w','m','cgl','cm','n','jp','vp','v','vg','vr','co','g','tv','k','o','an','tg','sp','asp','sci','int','out','toy','biz','i','po','p','ck','ic','wg','mu','fa','3','gd','diy','wsg','s','hc','hm','h','e','u','d','y','t','hr','gif','trv','fit','x','lit','adv','lgbt','mlp','b','r','r9k','pol','soc','s4s']
cache  = {cache: '' for cache in boards}

#Returns [ random image URL, random image's thread URL ]
def r4chan():
    #Select a board
    board = random.choice(boards)

    #Request board catalog, and get get a list of threads on the board; then sleeping for 1.5 seconds
    threadnums = list()
    data = ''

    #If a board's catalog has already been requested, just use that stored data instead
    if (cache[board] != ''):
        data = cache[board]
    #else request the catalog, and sleep for 1.5 seconds; storing that data for future use
    else:
        data = (requests.get('http://a.4cdn.org/' + board + '/catalog.json')).json()
        cache[board] = data
        time.sleep(1.5)

    #Get a list of threads in the data
    for page in data:
        for thread in page["threads"]:
            threadnums.append(thread['no'])

    #Select a thread
    thread = random.choice(threadnums)

    #Request the thread information, and get a list of images in that thread; again sleeping for 1.5 seconds
    imgs = list()
    pd = (requests.get('http://a.4cdn.org/' + board + '/thread/' + str(thread) + '.json')).json()
    for post in pd['posts']:
        #Ignore key missing error on posts with no image
        try:
            imgs.append(str(post['tim']) + str(post['ext']))
        except:
            pass
    time.sleep(1.5)

    #Select an image
    image = random.choice(imgs)

    #Assemble and return the urls
    imageurl = 'https://is2.4chan.org/' + board + '/' + image
    thread = 'https://boards.4chan.org/' + board + '/thread/' + str(thread)
    return [ imageurl , thread ]





# In[134]:


"""
A HANDFUL OF DATA TO TEST DATA TO TEST



analyze_image("https://static-ca-cdn.eporner.com/gallery/hc/sA/aR240AusAhc/997580-just-a-simple-nude-selfie-but-i-liked-it.jpg")


# In[133]:


analyze_image("https://fi1.ypncdn.com/201907/02/15422344/original/100(m=eKw7agaaaa).jpg")


# In[135]:


analyze_image("https://m.media-amazon.com/images/I/71kkMXAcLCL.png")


# In[136]:


analyze_image("https://images.unsplash.com/photo-1425082661705-1834bfd09dca?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxleHBsb3JlLWZlZWR8Mnx8fGVufDB8fHx8&w=1000&q=80")


# In[137]:


analyze_image("https://cdn77-pic.xvideos-cdn.com/videos/thumbs169poster/ee/3d/5f/ee3d5f02e7e0d5278a15bc304603a15e/ee3d5f02e7e0d5278a15bc304603a15e.30.jpg")


# In[138]:


analyze_image("https://static-ca-cdn.eporner.com/thumbs/static4/5/59/595/5955465/4_240.jpg")


# In[139]:


analyze_image("https://external-preview.redd.it/eO9IxAkN_8CXx9cRpGDuuNxbZ7jqJNfgqCBFc0bBVkQ.jpg?width=640&crop=smart&auto=webp&s=2e43781e5b75af5845586c7bd75728518554761f")


# In[140]:


analyze_image("https://iv1.lisimg.com/image/24104978/540full.jpg")


# In[141]:


analyze_image("https://i0.nekobot.xyz/8/4/3/ee6b349d9951c6a546ad79bc79fe69376290b.jpg")

"""
# In[148]:



"""
HERE GOES THE FLASK FUNCTIONS

"""




app = Flask(__name__)

@app.route("/retrieve")
def retrieve():
    element = analyze_image(r4chan()[0])

 
    return json.dumps({"link": element[0], "prediction": element[1][0]})
   
        
    
if __name__ == "__main__":
    app.run(debug=True)
    
    







