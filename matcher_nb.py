from sklearn.naive_bayes import GaussianNB
import numpy as np

def m_Np(X, y):

    nb = GaussianNB() 
    num_correct = 0
    gen_scores = []
    num_incorrect = 0
    imp_scores = []
    
    for i in range(0, len(y)):
        query_img = X[i, :]
        query_label = y[i]
        
        template_imgs = np.delete(X, i, 0)
        template_labels = np.delete(y, i)
            
        # Set the appropriate labels
        # 1 is genuine, 0 is impostor
        y_hat = np.zeros(len(template_labels))
        y_hat[template_labels == query_label] = 1 
        y_hat[template_labels != query_label] = 0
        
        nb.fit(template_imgs, y_hat) # Train the classifier
        y_pred = nb.predict(query_img.reshape(1,-1)) # Predict the label of the query
        
        # Print results
        if y_pred == 1:
            num_correct += 1
            gen_scores.append(query_label)
        else:
            num_incorrect += 1
            imp_scores.append(query_label)
            
    return gen_scores, imp_scores
    # Print results
    #print()
    #print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
          #% (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect)))    
    
    