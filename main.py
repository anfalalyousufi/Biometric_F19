import features_lbp
import features_pca
import features_landmarks
import matcher_nb
import matcher_knn
import performance
import get_data

feature = 1
matcher = 2

''' Load the data and their labels '''
image_directory = 'Project Data'
landmark_directory = 'Project Data/landmarks'
X, y = get_data.get_images(image_directory)


if feature == 1:
    X = features_pca.pca(X)
elif feature == 2:
    X, y = features_landmarks.get_landmarks(landmark_directory)
elif feature == 3:
    X = features_lbp.lbpfeature(X)
    
if matcher == 1:
    gen_scores, imp_scores = matcher_nb.m_Np(X, y)
elif matcher == 2:
    gen_scores, imp_scores = matcher_knn.knn(X, y)

''' Performance assessment '''
performance.perf(gen_scores, imp_scores)

