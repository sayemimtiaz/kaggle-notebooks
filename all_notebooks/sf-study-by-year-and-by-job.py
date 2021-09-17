# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load in 
# 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting lib
# 
# # Input data files are available in the "../input/" directory.
# # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# 
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# 
# # Any results you write to the current directory are saved as output.
# 
df = pd.read_csv("../input/Salaries.csv");

# 
#df.dtypes
#df.describe()
#df


# # sort by salaries
df.iloc[3:5,0:7]
#df.sort_index(axis=1, ascending=False);
#df.sort_values(by='TotalPay');



plt.figure()
df2 = df.loc[:,['TotalPay', 'JobTitle', 'BasePay']].sort_values(by="TotalPay", ascending=False);

df





df3=df.loc[:,['TotalPay', 'JobTitle']]
df3
dfp = df3.groupby('JobTitle').mean();
#dfp = dfp.sort_values(by='TotalPay', ascending=False)
dfp=dfp.ix[1:20]
plt1 = dfp.plot(kind='bar');

print(plt1)
df3=df.loc[:,['TotalPayBenefits', 'JobTitle']]
df3
dfp = df3.groupby('JobTitle').mean();
dfp=dfp.ix[1:25]
plt2 = dfp.plot(kind='bar');

print(plt2)