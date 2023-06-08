import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


training = 'C:\\Users\\cole_\\OneDrive\\Desktop\\Uni of Southampton\\Semester 6 redo\\Security of Cyber Physical Systems\\cw2\\TrainingDataBinary.csv'
testing = 'C:\\Users\\cole_\\OneDrive\\Desktop\\Uni of Southampton\\Semester 6 redo\\Security of Cyber Physical Systems\\cw2\\TestingDataBinary.csv'

# Step 1: Data reading
trainingCSV = pd.read_csv(training)
testingCSV = pd.read_csv(testing)

# Step 2: Reading CSV file
yTrain = trainingCSV['Attack'].values
xTrain = trainingCSV.drop(labels=['Attack'], axis=1)
xTest = testingCSV.drop(labels=['Attack'], axis=1)

#print("xTrain: ", xTrain.shape)
#print("xTest: ", xTest.shape)
#print(xTrain.head())
#print(xTest.head())

pca = PCA()
x_pca = pca.fit_transform(xTrain)
print(xTrain.shape)
print(x_pca.shape)

logreg = LogisticRegression(C=21, max_iter=15000)
logreg2 = LogisticRegression(C=21, max_iter=15000)


print("\nLogisticRegression score: %f" % logreg.fit(xTrain, yTrain).score(xTrain, yTrain))
#print("\nLogisticRegression score PCA: %f" % logreg.fit(x_pca, yTrain).score(xTrain, yTrain))

predict_xTest = logreg.predict(xTest)
print(predict_xTest)

prediction_test = logreg.predict(xTrain)
#print(prediction_test)
print("Accuracy: ", metrics.accuracy_score(yTrain, prediction_test))

#print(f1_score(yTrain, prediction_test, average='macro'))


"""
#clf = svm.SVC(kernel='linear', C=1e5).fit(xTrain, yTrain)
#print(clf.score(xTrain, yTrain))
"""

#scores = cross_val_score(logreg.fit(xTrain, yTrain), xTrain, yTrain, n_jobs=2, cv=10)
#print(scores.get_params())

"""
scaler = StandardScaler()
pipe = Pipeline(steps=[("scaler", scaler), ("pca", pca), ("logistic", logreg2)])

param_grid = {
    "pca__n_components": [10, 30, 50, 70, 90, 110, 120, 128],
    "logistic__C": np.logspace(-1, 1, 1),
}

scoresGrid = GridSearchCV(pipe, param_grid, cv=10)
scoresGrid.fit(xTrain, yTrain)
print("Best parameter (CV score=%0.3f):" % scoresGrid.best_score_)
print(scoresGrid.best_params_)
"""

cm = confusion_matrix(yTrain, prediction_test, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()
plt.show()