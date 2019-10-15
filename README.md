# Biometric_F19
In this project, you will work with your group members to develop a face recognition system thatwill be tested on the images captured from the team. From the folder FaceData, I havepreprocessed the images with a face detector. As you know, some of the images exhibit morevariation compared to others in factors such as lighting, pose, facial expressions, distance fromthe camera, and occlusions. You have three goals for this project:  <br/>
1.Build a face recognition system. <br/>
2.Analyze the system against images with varying amounts of intra-class variation. <br/>
3.Describe the performance of the system.<br/><br/> 

## To do

To accomplish these goals, you will need to complete the following tasks:<br/>
1.Starting from Homework 2, edit the code to also extract features using PCA and LBP. Atthe completion of this task, you should have three independent modules (i.e., .py scripts)that return a dataset of PCA features, LBP features, and landmark features.<br/>
2.Edit the code further to use k-Nearest Neighbors and Naive Bayes classifiers. At thecompletion of this task, you should have two independent modules (i.e., .py scripts) thatreturn the results for the respective classifier.<br/>
3.Write a main.py script that allows you to set the parameters for the system (i.e., featuresand matcher to use).<br/>
4.Use the script from Homework 4 to analyze the performance of the system based on theconditions provided in main.py.<br/><br/>

At the completion of these tasks, you should have the following <br/>
files:<br/>main.py<br/>
features_lbp.py<br/>
features_pca.py<br/>
features_landmarks.py<br/>
matcher_knn.py<br/>
matcher_nb.pyperformance.py<br/>
