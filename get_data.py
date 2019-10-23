# Load imports
import os 
import cv2
import numpy as np
import re
from PIL import Image
from PIL import ImageEnhance


def get_images(image_directory,flag):
    X = []
    y = []
    extensions = ('jpg','png','gif')
    
    '''
    Each subject has their own folder with their
    images. The following line lists the names
    of the subfolders within image_directory.
    '''
    subfolders = os.listdir(image_directory)
    for subfolder in subfolders:
        print("Loading images in %s" % subfolder)
        if os.path.isdir(os.path.join(image_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(image_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith(extensions): # grab images only
                    # read the image using openCV
                    img = cv2.imread(
                            os.path.join(image_directory, subfolder, file), cv2.IMREAD_GRAYSCALE
                            )
                    '''
                    Here we are trying to split file name to get the task number.
                    We use if and else when flag is set because 2 members of group have tasks labeled as - for example - 1_0.png
                    and others have Task1_0.png.
                    We brighten only the dark images in tasks 6-10
                    '''
                    '''
                    #Flag is 1 for system 2
                    if flag==1:
                        tokens=[]   #to hold initial split of filename stored in file variable for images labelled as starting with 'Task'
                        numbers=[]  #to hold initial split of filename stored in file variable for images not labelled as starting with 'Task'
                        if file.startswith('Task'):  #for images labelled starting with Task, we split file name twice 
                            print(file)
                            if re.findall("/6|7|8|9/",file) or re.findall("10",file): #using regular expression to check if we encounter task 6-10 images
                                tokens.append(file.split('k')[1])
                                number=tokens[0].split('_')[0]
                                filenum=int(number[0])  #convert string number to int
                                if filenum> 5 and filenum <11:  #since the regular expression returns 16 and 6 when searching for 6, we retain task numbers greater than 5 and less than 11
                                    img = Image.fromarray((img).astype(np.uint8)) #working on actual image
                                    enhance_b=ImageEnhance.Brightness(img)
                                    brightness=20.0
                                    img=enhance_b.enhance(brightness)   #brighten the image by a factor of 20                           
                                    img=np.array(img) #converting image back to numpy array
                                    #img=img+20
                        else:
                            if re.findall("/6|7|8|9/",file) or re.findall("10",file): #for images labelled not starting with Task, we split file name once 
                                numbers.append(file.split('_')[0])
                                filenum=int(numbers[0])
                                if filenum> 5 and filenum <11:
                                    img = Image.fromarray((img).astype(np.uint8))
                                    enhance_b=ImageEnhance.Brightness(img)
                                    brightness=20.0
                                    img=enhance_b.enhance(brightness)                              
                                    img=np.array(img)
                                    #img=img+20
                    else:
                        continue
                    '''
                    
                    # resize the image by half
                    scale_percent = 50 # percent of original size
                    width = int(img.shape[1] * scale_percent / 100)
                    height = int(img.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    img = cv2.resize(img, dim)
                    
                    # add the resized image to a list X
                    X.append(img)
                    # add the image's label to a list y
                    y.append(subfolder)
    
    print("All images are loaded")     
    # return the images and their labels      
    return np.array(X), np.array(y)          