# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import seaborn
import seaborn as sns

# Apply the default theme
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
