# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 16:15:09 2023

@author: 3ndalib
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pylab as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score,recall_score
df = pd.read_csv("E:\Projects\ElectroPiExtras\Social_Network_Ads.csv")
# print(df)
df=df.drop("User ID",axis=1)
# print(df)
LE = LabelEncoder()
df["Gender"] = LE.fit_transform(df["Gender"])

# print(df)
#sns.pairplot(df,hue="Purchased")
x = df.drop("Purchased",axis=1)
y = df["Purchased"]
x = StandardScaler().fit_transform(x)
XTrain,XTest,YTrain,YTest=tts(x,y,test_size=0.2,shuffle=True)
clsf = KNeighborsClassifier()
clsf.fit(XTrain, YTrain)
preds = clsf.predict(XTrain)
TestPreds = clsf.predict(XTest)
# print(accuracy_score(YTrain, preds))
# print(recall_score(YTrain, preds))

MaxAge,MinAge = x[:1].max()-1,x[:1].min()-1
MaxSalary,MinSalary = x[:2].max()-1,x[:2].min()-1

AgeGrid,SalaryGrid = np.meshgrid(np.arange(MinAge,MaxAge,0.02),np.arange(MinSalary,MaxSalary,0.02))
# print("HelloThere")


# for k in range(1,10,2):
#     clsf =KNeighborsClassifier(n_neighbors=k)
#     clsf.fit(XTrain, YTrain)
#     preds = clsf.predict(XTrain)
#     TestPreds = clsf.predict(XTest)
#     print(k)
#     print("Train acc",accuracy_score(YTrain, preds))
#     print("Test acc",accuracy_score(YTest, TestPreds))
    
plt.figure(figsize=(30,30))
plt.set_cmap(plt.cm.cool_r)

i=1

for k in range(1,10,2):
    clsf =KNeighborsClassifier(n_neighbors=k)
    clsf.fit(XTrain[:,1:], YTrain)
    preds = clsf.predict(XTrain[:,1:])
    TestPreds = clsf.predict(XTest[:,1:])
    print(k)
    print("Train acc",accuracy_score(YTrain, preds))
    print("Test acc",accuracy_score(YTest, TestPreds))

    plt.subplot(5, 2, i)
    z= clsf.predict(np.c_[AgeGrid.ravel(),SalaryGrid.ravel()])
    z = z.reshape(AgeGrid.shape)
    plt.set_cmap(plt.cm.cool_r)
    plt.contourf(AgeGrid,SalaryGrid, z)
    plt.axis("tight")
    plt.scatter(XTrain[:,1], XTrain[:,2], c=YTrain)
    plt.title(f"for iter {k} train")

    i+=1
    plt.subplot(5, 2, i)
    plt.set_cmap(pl.cm.cool_r)
    plt.contourf(AgeGrid,SalaryGrid, z)
    plt.axis("tight")
    plt.scatter(XTest[:,1], XTest[:,2], c=YTest)
    plt.title(f"for iter {k} test")
