# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Cleaned_Dataset.xlsx')
X = dataset.iloc[:, :-1].values      # Features
y = dataset.iloc[:, -1].values       # Labels

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 19, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy on test set is: ',accuracy_score(y_test, y_pred))

# Validating model accuracy on train set

from sklearn.model_selection import cross_val_score
True_accuracy=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
True_accuracy=True_accuracy.mean()*100
print(f'True model accuracy is: ',True_accuracy)

# Finding best 'n_estimator' parameter for  model

from sklearn.model_selection import GridSearchCV
parameter=[{'n_estimators':[3,5,7,9,11,13,15,17,19]}]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameter,scoring='accuracy',cv=10,n_jobs=-1)
grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_

print(f'Best accuracy is: ',best_accuracy)
print(f'Best parameter is: ',best_parameters)