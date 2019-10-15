import get_data
import get_features
import matcher
import performance 

''' Load the data and their labels '''
image_directory = 'data'
X, y = get_data.get_images(image_directory)

#landmark_directory = 'data/landmarks'
#X, y = get_data.get_landmarks(landmark_directory)

''' Get PCA components '''
X = get_features.pca(X)

''' Matching with knn'''
gen_scores, imp_scores = matcher.knn(X, y)

''' Performance assessment '''
performance.perf(gen_scores, imp_scores)

