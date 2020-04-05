import numpy as np
import scipy.io as scio
import sklearn.svm as skvm

from ProcessEmail import ProcessEmail

#.....Loading and processing e-mail.....
email=open('emailSample1.txt')
email1=email.read()

print("\nOriginal email :\n\n",email1,"\n")
print("processing e-mail......")

word_index,vocab_dict=ProcessEmail(email1)
print("\nWord Indices : \n\n",word_index)

#.....Feature Extraction.....
features=np.zeros([len(vocab_dict),1])
for i in word_index:
    features[i]=1

#.....Training with SVM.....
data1=scio.loadmat('spamTrain.mat')
X_train,y_train=data1['X'],data1['y']
y_train=y_train.ravel()
count=0

print("\nTraining Spam Classifier........")
svm=skvm.SVC(C=0.1,kernel='linear')
model=svm.fit(X_train,y_train)
print("...Done!")

#.....Prediction on Train Data.....
pred=model.predict(X_train)
for i in range(len(y_train)):
    if pred[i]==y_train[i]:
        count+=1
print("Train Accuracy : ",count*100/len(y_train))

#.....Loading and Predicting Test Data.....
data2=scio.loadmat('spamTest.mat')
X_test,y_test=data2['Xtest'],data2['ytest']
count=0

pred=model.predict(X_test)
for i in range(len(y_test)):
    if pred[i]==y_test[i]:
        count+=1
print("Test  Accuracy : ",count*100/len(y_test))