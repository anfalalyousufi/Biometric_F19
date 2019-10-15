# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn.datasets import fetch_olivetti_faces

width = 64
dataset = fetch_olivetti_faces(shuffle=True)
faces = dataset.data
n_samples, n_features = faces.shape
print("Dataset consists of %d faces" % n_samples)

plt.figure()
plt.imshow(faces[0].reshape((width,width)), cmap='gray')

# Squash it
faces = faces.transpose() # each column is a squashed face

# Get the mean face
mean_face = faces.mean(axis=1)
plt.figure()
plt.imshow(mean_face.reshape((width,width)), cmap='gray')


# Subtract the mean face - center everybody
for col in range(faces.shape[1]):
    faces[:,col] = faces[:,col] - mean_face
    
plt.figure()
plt.imshow(faces[:,0].reshape((width,width)), cmap='gray')


# Compute the covariance matrix, C
C = np.cov(faces.transpose())


# Get the eigenfaces from C
evals, evecs = np.linalg.eig(C)

# Show some eigenfaces
eigenfaces = np.dot(faces, evecs)
plt.figure()
plt.subplot(131)
plt.imshow(eigenfaces[:,0].reshape(width, width), cmap='gray')
plt.subplot(132)
plt.imshow(eigenfaces[:,1].reshape(width, width), cmap='gray')
plt.subplot(133)
plt.imshow(eigenfaces[:,2].reshape(width, width), cmap='gray')


# Can we really reconstruct the face?
k = 20
face1 = np.zeros(width**2)
for i in range(k):
    face1 += eigenfaces[:,i].transpose() * faces[:,0] * eigenfaces[:,i]    
plt.figure()
plt.imshow(face1.reshape(width, width), cmap='gray')

# Finally, what do the features look like?
face_features = np.zeros((faces.shape[1], k))
for i in range(faces.shape[1]):
    face = faces[:,i]
    for j in range(k):
        face_features[i,j] = np.dot(eigenfaces[:,j].transpose(), face)



