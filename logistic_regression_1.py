#importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
#importing dataset
clean_data=pd.read_csv('clean_data.csv')
clean_data = clean_data.drop(['Unnamed','Unnamed: 0','Unnamed: 0.1'], axis=1)
X=clean_data.iloc[:,5:11]
Y=clean_data.iloc[:,0]

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#predicting training set results and thus finding accuracy
y_pred_train=classifier.predict(X_train)
#Making confusion matrix
cm_train=confusion_matrix(Y_train,y_pred_train)
plt.figure(figsize=(20,10))
sn.heatmap(cm_train, annot=True)
plt.show()
correct_train=0
for i in range (0,8):
    correct_train=correct_train+cm_train[i,i]
accuracy_train=(correct_train/Y_train.size)*100
print('TRAINING ACCURACY IS ',accuracy_train)
#predicting test set results and thus finding accuracy
y_pred_test=classifier.predict(X_test) 
#Making Confusion Matrix
cm_test=confusion_matrix(Y_test,y_pred_test)
plt.figure(figsize=(20,10))
sn.heatmap(cm_test, annot=True)
plt.show()
correct_test=0
for i in range (0,8):
    correct_test=correct_test+cm_test[i,i]
accuracy_test=(correct_test/Y_test.size)*100
print('TEST ACCURACY IS ',accuracy_test)
