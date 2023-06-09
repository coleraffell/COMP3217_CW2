import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

# Getting the CSV files
training = 'C:\\Users\\cole_\\OneDrive\\Desktop\\Uni of Southampton\\Semester 6 redo\\Security of Cyber Physical Systems\\cw2\\TrainingDataMulti.csv'
testing = 'C:\\Users\\cole_\\OneDrive\\Desktop\\Uni of Southampton\\Semester 6 redo\\Security of Cyber Physical Systems\\cw2\\TestingDataMulti.csv'

# Step 1: Data reading
trainingCSV = pd.read_csv(training)
testingCSV = pd.read_csv(testing)
xTrain = trainingCSV.drop(labels=['Attack'], axis=1)
yTrain = trainingCSV['Attack'].values
xTest = testingCSV.drop(labels=['Attack'], axis=1)

# Two different logistic regression instances for displaying score with tuned hyperparameter

lm = LogisticRegression(multi_class='multinomial', max_iter=10000, C=21)
lmFit = lm.fit(xTrain, yTrain)
print("\nLogisticRegression score: %f" % lm.fit(xTrain, yTrain).score(xTrain, yTrain))


#logreg = LogisticRegression(C=21, max_iter=15000)
#logreg2 = LogisticRegression(max_iter=15000)

# Attempting to use dimension reduction
#pca = PCA()
#x_pca = pca.fit_transform(xTrain)
#print("\nLogisticRegression score PCA: %f" % logreg.fit(x_pca, yTrain).score(xTrain, yTrain))

# Displaying logistic regression scores to compare hyperparameter tuning to no tuning
#print("\nLogisticRegression score: %f" % logreg2.fit(xTrain, yTrain).score(xTrain, yTrain))
#print("\nLogisticRegression score hyperparameters: %f" % logreg.fit(xTrain, yTrain).score(xTrain, yTrain))

# Attempting support vector machine SVM method
#clf = svm.SVC(kernel='linear', C=21).fit(xTrain, yTrain)
#print(clf.score(xTrain, yTrain))

# Performing some cross validation
#scores = cross_val_score(logreg.fit(xTrain, yTrain), xTrain, yTrain, n_jobs=2, cv=10)
#print(scores)

# Trying to use Dimension Reduction
#scaler = StandardScaler()
#pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logreg2)])

"""
param_grid = {
    "C": [20, 20.3, 20.6, 20.9, 21.2, 21.5],
    "pca__n_components": [10, 30, 50, 70, 90, 110, 120, 128],
    "logistic__C": np.logspace(-1, 1, 1),
}

scoresGrid = GridSearchCV(pipe, param_grid, cv=10)
scoresGrid.fit(xTrain, yTrain)
print("Best parameter (CV score=%0.3f):" % scoresGrid.best_score_)
print(scoresGrid.best_params_)
"""

# Trying to use Standardardization
pipe = make_pipeline(StandardScaler(), LogisticRegression(C=21, max_iter=10000))
pipe.fit(xTrain, yTrain)
pipeline = Pipeline(steps=[('standardscaler', StandardScaler()), ('logisticregression', LogisticRegression())])
print("Standardised Score: ", pipe.score(xTrain, yTrain))

# Predicting the test results
#prediction_test = pipe.predict(xTest)
#print(prediction_test)

# Confusion Matrix
cm = confusion_matrix(yTrain, pipe.predict(xTrain), labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.show()
