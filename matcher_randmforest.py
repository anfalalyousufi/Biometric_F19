'''
Matcher for all the systems
'''

from sklearn.ensemble import RandomForestClassifier
import numpy as np


def RF(X, y):
    #random forest with 10 trees and maximum depth of tree is set to 25. 
    #We choose to specify a depth to avoid trees from getting too big and thus overfitting
    randomFor = RandomForestClassifier(n_estimators=10, max_depth=25) 
    gen_scores = []
    imp_scores = []
    
    for i in range(0, len(y)):
        query = X[i, :]
        query_label = y[i]
        
        templates = np.delete(X, i, 0)
        template_labels = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_labels))
        y_hat[template_labels == query_label] = 1 
        y_hat[template_labels != query_label] = 0
        
        randomFor.fit(templates, y_hat) # Train the classifier
        scores = randomFor.predict_proba(query.reshape(1,-1)).reshape(1,2) # Predict the probabilistic label of the query
        classes = randomFor.classes_.reshape(1,2)
        
        gen_scores.extend(scores[classes==1])
        imp_scores.extend(scores[classes==0])
        
    return gen_scores, imp_scores