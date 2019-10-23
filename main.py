import features_lbp
import features_pca
import matcher_nb
import matcher_knn
import performance
import get_data
import RF
import enhancement
import numpy as np

'''
Systems: 
    1. Random Forest with PCA
    2. Random Forest with PCA and Image Enhancement
    3. Random Forest with Feature Level Fusion of PCA and LBP
'''
system = 2

#Load the data and their labels
image_directory = r"C:\Users\Mary\Desktop\Biometric Auth Project\ProjectData"
landmark_directory = r"C:\Users\Mary\Desktop\Biometric Auth Project\ProjectData"

#The get_data call is in this function so we can separate image enhancement.
if system == 1:
    X, y = get_data.get_images(image_directory)
    
    #PCA
    X = features_pca.pca(X)
    
    #Random Forest
    gen_scores, imp_scores = RF.RF(X, y)
    
elif system == 2:
    #Image enhancement
    X, y = enhancement.get_images(image_directory)
    
    #PCA
    X = features_pca.pca(X)
    
    #Random Forest
    gen_scores, imp_scores = RF.RF(X, y)
    
elif system == 3:
    X, y = get_data.get_images(image_directory)
    
    #Feature level fusion
    W = Z = X
    W = features_pca.pca(W)
    Z = features_lbp.featuresLBP(Z)
    X = np.hstack((W, Z))
    
    #Random Forest
    gen_scores, imp_scores = RF.RF(X, y)
    
#Performance assessment
performance.perf(gen_scores, imp_scores)


#Old main layout in case we need it

#features: 1 = PCA, 2 = LBP, 3 = Fusion
#matchers: 1 = NB, 2 = KNN, 3 = RF

''' 
if feature == 1:
    X = features_pca.pca(X)
elif feature == 2:
    X = features_lbp.featuresLBP(X)
elif feature == 3:
    W = Z = X
    W = features_pca.pca(W)
    Z = features_lbp.featuresLBP(Z)
    X = np.hstack((W, Z))
    
if matcher == 1:
    gen_scores, imp_scores = matcher_nb.m_Np(X, y)
elif matcher == 2:
    gen_scores, imp_scores = matcher_knn.knn(X, y)
elif matcher == 3:
    gen_scores, imp_scores = RF.RF(X, y)
'''