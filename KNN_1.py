#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import  accuracy_score,classification_report,confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='YlOrRd'):
      
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(20,10),)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout();

#importing dataset
clean_data=pd.read_csv('clean_data.csv')
clean_data = clean_data.drop(['Unnamed','Unnamed: 0','Unnamed: 0.1'], axis=1)
X=clean_data.iloc[:,5:11]
Y=clean_data.iloc[:,0]
activities=["climbingdown","climbingup","jumping","lying","running","sitting","standing","walking"]
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#fitting logistic regression
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_train,Y_train)
#predicting training set results and thus finding accuracy
y_pred_train=classifier.predict(X_train)
#Making confusion matrix
cm_train=confusion_matrix(Y_train,y_pred_train)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm_train, 
                      classes=activities,
                      title='');
print(classification_report(Y_train, y_pred_train))
print(f'Training acc score is: {accuracy_score(Y_train,y_pred_train)*100:.3f}')
#predicting test set results and thus finding accuracy
y_pred_test=classifier.predict(X_test) 
#Making Confusion Matrix
cm_test=confusion_matrix(Y_test,y_pred_test)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm_test, 
                      classes=activities,
                      title='');
print(classification_report(Y_test, y_pred_test))
print(f'Test acc score is: {accuracy_score(Y_test,y_pred_test)*100:.3f}')