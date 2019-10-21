import features_lbp
import features_pca
import matcher_nb
import matcher_knn
import performance
import get_data
import RF
import numpy as np

feature = 3 #1 = PCA, 2 = LBP, 3 = Fusion
matcher = 3 #1 = NB, 2 = KNN, 3 = RF

''' Load the data and their labels '''
image_directory = r"C:\Users\Mary\Desktop\test"
landmark_directory = r"C:\Users\Mary\Desktop\test"
X, y = get_data.get_images(image_directory)

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
    
''' Performance assessment '''
performance.perf(gen_scores, imp_scores)