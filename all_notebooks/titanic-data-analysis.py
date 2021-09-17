# %% [code]
#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# %% [markdown]
# Loading Data

# %% [code]
train=pd.read_csv("../input/train.csv")
train.head()


# %% [markdown]
# Load test data

# %% [code]
test=pd.read_csv("../input/test.csv")
test.head()

# %% [markdown]
# Survived plot

# %% [code]
train['Survived'].value_counts().plot('bar')


# %% [markdown]
# We have seen that number of dead people are alomst double number of survived people

# %% [markdown]
# Percentage and missing number of data per column 

# %% [code]
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# %% [markdown]
# as we can see that cabin data is almost unusful due to huge about of missing values , so we can neglict it 

# %% [markdown]
# Drop the Cabin column from the dataset

# %% [code]
train = train.drop(columns=['Cabin'])


# %% [markdown]
# ****reset the missing values of Age column with the mean 

# %% [code]
train.Age=train.Age.fillna(train.Age.mean())


# %% [markdown]
# We gonna print the column types 

# %% [code]
train.dtypes

# %% [code]
sns.violinplot(x="Survived", y="Pclass", data=train)
plt.show()

# %% [markdown]
# as we can see that class 1 has the most survived number of people, but calss 3 has the most dead number of people

# %% [code]
sns.violinplot(x="Survived", y="Age", data=train)
plt.show()

# %% [markdown]
# as we can see that, most dead people are between 20 to 30 years old, and at age between 0 and 10 we have more survivers 

# %% [code]
sns.violinplot(x="Survived", y="Fare", data=train)
plt.show()

# %% [markdown]
# we can see that most of the dead people paied low fare

# %% [markdown]
# We gonna plot frequncey table between Survivers and other parameters

# %% [code]
pd.crosstab(train.Survived,train.Sex)

# %% [code]
pd.crosstab(train.Survived,train.SibSp)

# %% [code]
pd.crosstab(train.Survived,train.Parch)

# %% [code]
pd.crosstab(train.Survived,train.Embarked)