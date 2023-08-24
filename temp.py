# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'


df = pd.read_csv("churn-bigml-80.csv")
pd.options.display.max_columns = None
sns.set_theme()

# Load an example dataset
tips = pd.read_csv("BankChurners.csv")

# Create a visualization
print(tips.head())


# fig, axis= plt.subplots(1,2)
# df = pd.read_csv("BankChurners.csv")
# axis[0].hist(df["Customer_Age"],bins=10)
# axis[1].hist(df["Customer_Age"],bins=10,cumulative=True)


# fig, axis= plt.subplots(2,1)

# df = pd.read_csv("BankChurners.csv")
# count = df["Gender"].value_counts()
# count_df = pd.DataFrame(count)
# print(count)
# axis[0].bar(count_df.T.columns, count)
# axis[1].barh(count_df.T.columns, count)
# male = df[df["Gender"]=="M"]
# female = df[df["Gender"]=="F"]


#x = np.linspace(0, 10,100)


# fig , axis = plt.subplots(1, 3)

# axis[0].plot(x,x,linewidth = 3,label="Linear")
# axis[1].plot(x,x**2, linestyle ="--",label="Square")
# axis[2].plot(x,x**3,label="Cube")
# fig , axis = plt.subplots(2, 3)
# fig , axis = plt.subplots(2, 2)

# axis[0][0].plot(x,x,linewidth = 3,label="Linear")
# axis[0][1].plot(x,x**2, linestyle ="--",label="Square")
# axis[1][0].plot(x,x**3,label="Cube")
# fig.legend(loc="upper left")

# plt.xlabel("XAxis")
# plt.ylabel("YAxis")
# plt.title("X vs X Square")
# plt.show()
print(df.info())
# print(df.head())
# print(df.shape)
# print("--------------- DataTypes ---------------")
# print(df.dtypes)
# print(df.columns[0])
# print(df[df.columns[0]])
# print(df.describe())
# print( df.describe( include=[ "object" ,"bool"] ) )
# df["Churn"] = df["Churn"].astype("int")
# print(df["Churn"].value_counts())
# print(df["Churn"].value_counts(normalize=True))
# print(df.sort_values(by = "Total day calls", ascending=False))
# print(df.sort_values(by = ["Total day calls"
#                            ,"Total night calls"], ascending=[False,False]))

# print(df[df.columns[1]].mean())
# print(df[df["Churn"]==1]["Total day minutes"])
# print(df[(df["Churn"]==1)&(df["International plan"]=="No")]["Total day minutes"])


# print(df.loc[:5, ["Churn","Total day minutes"]]               )
# print(df.iloc[:5, [0,1]]               )
# print(df.iloc[:5, :5]               )
# print(df.apply(np.max))
# print(df[df["State"].apply(lambda x: x[0] == "M")])
# print(df[df[df.columns[0]].apply(lambda x: x[0] == "W")])
# d ={"Yes":True,"No":False}
# df["International plan"]=df["International plan"].map(d)
# print(df["International plan"])
# df = df.replace({"Voice mail plan":d})
# print(df["Voice mail plan"])
# print(pd.crosstab(df["Churn"], df["International plan"]))
# print( df.pivot_table(["Voice mail plan","International plan",
#                        "Area code"],["Churn"], aggfunc="sum")       )
# sns.boxplot(df[df.columns[1]].head())
# sns.violinplot(df[df.columns[1]].head())
# plt.hist(df[df.columns[0]].head(),rwidth=0.7)
# plt.hist(df[df.columns[1]])
# print(df.isnull().any(axis=1).sum())
# df["Total Calls"] = 500
# print(df)
# df = df.drop("Total Calls",axis=1)
# print(df)
# df[df.columns[1]].plot(kind="density")
# plt.hist(df[df.columns[1]])
# --------------------univariate categorical--------------------
# sns.countplot(x=df.columns[0],data=df)
#print(pd.DataFrame(df[df.columns[0]].value_counts()).T.columns)
#print(df[df.columns[0]].value_counts())
#plt.scatter(df[df.columns[0]], df[df.columns[0]].value_counts())
#plt.bar(pd.DataFrame(df[df.columns[0]].value_counts()).T.columns, df[df.columns[0]].value_counts())
#plt.scatter(pd.DataFrame(df[df.columns[0]].value_counts()).T.columns, df[df.columns[0]].value_counts())
# plt.pie(df[df.columns[0]].value_counts(),labels=pd.DataFrame(df[df.columns[0]].value_counts()).T.columns)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

newdf = df.select_dtypes(include=numerics)
sns.heatmap(newdf.corr())
plt.legend()

plt.show()
# ------------------------------------------
# for column in range( df.shape[1]):
#     if(ObjectDtype(df[df.columns[column]])):
#         print(df.columns[column])
#         plt.figure()
#         plt.hist(df[df.columns[column]], rwidth=0.9)
#         plt.title(df.columns[column])
#         plt.legend()
#         plt.show()
#     else:
#         print("No")
#         print(df.columns[column])

# -------Multivariate-------
