import os 
import numpy as np

def get_landmarks(landmark_directory):
    X = []
    y = []
    
    subfolders = os.listdir(landmark_directory)
    for subfolder in subfolders:
        print("Loading landmarks in %s" % subfolder)
        if os.path.isdir(os.path.join(landmark_directory, subfolder)): # only load directories
            subfolder_files = os.listdir(
                    os.path.join(landmark_directory, subfolder)
                    )
            for file in subfolder_files:
                if file.endswith('5.npy'): 
                    landmarks = np.load(os.path.join(landmark_directory, subfolder, file))
                    X.append(landmarks)
                    y.append(subfolder)
    
    print("All landmarks are loaded")     
    # return the images and their labels      
    return np.array(X), np.array(y)