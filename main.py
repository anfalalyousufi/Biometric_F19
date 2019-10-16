import features_lbp
import features_pca1
import features_landmarks
import matcher_nb
import matcher_knn
import performance
import get_images

features = [1,2,3]
matchers = [1,2]
#data = load_images()

''' Load the data and their labels '''
image_directory = '/Users/meghna/Desktop/Project 1/Project Data'
X, y = get_images.get_images(image_directory)

for feature in features:
    if feature == 1:
        pca_X = features_pca1.pca(X)
    if feature == 2:
        lndmrk_X, lndmrk_y = features_landmarks(X)
    if feature == 3:
        lbp_X, lbp_y = features_lbp(X)

#feature-level fusion
for i,j in range(0,len(pca_X)):
    X[i],y[i]=avg(pca_X[i],lndmrk_X[i],lbp_X[i]),avg(pca_y[i],lndmrk_y[i],lbp_y[i])
    
    for matcher in matchers:
        if matcher == 1:
            genuine_scores, impostor_scores = matcher_nb(X, y)
        if matcher == 2:
            genuine_scores, impostor_scores = matcher_knn(X, y)
            
#score-level fusion
        
        performance(genuine_scores, impostor_scores)
