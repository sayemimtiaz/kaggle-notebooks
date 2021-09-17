# %% [markdown]
# **Problem solution -1**

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

davis = pd.read_csv("../input/davis-data-set/Davis.csv")

davis.columns = ['id','sex', 'weight', 'height', 'repwt', 'repht']
davis.sex = pd.get_dummies(davis['sex'])

print(davis.head())
print()
print(davis.weight.corr(davis.height))
print(davis.weight.corr(davis.sex))

ax = plt.gca()
ax.bar(davis.weight, davis.repwt, align='edge')
ax.set_xlabel('weight')
ax.set_ylabel('repwt')
plt.show()
sns.set(style = 'whitegrid', context = 'notebook')
cols = ['sex','height', 'weight']
sns.pairplot(davis[cols],height = 2.5)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
davis.head()
X = davis.loc[:,['sex','height']].values
Y = davis['weight'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

slr = LinearRegression()
slr = slr.fit(X_train, Y_train)
y_train_pred = slr.predict(X_train)


y_test_pred = slr.predict(X_test)
print('R2 training: %.3f, test: %.3f' %(r2_score(Y_train, y_train_pred),r2_score(Y_test,y_test_pred)))

x_davis_hght = davis.height.values
x_davis_hght = np.resize(x_davis_hght, (200,1))
y_train_pred_graph = np.resize(y_train_pred,(200,1))
y_test_pred_graph = np.resize(y_test_pred,(200,1))

plt.scatter(x_davis_hght, y_train_pred_graph, c = 'blue',marker ='o', label = 'Train Data')
plt.scatter(x_davis_hght,y_test_pred_graph, c = 'green', marker = 's', label ='Test Data')
plt.xlabel('height')
plt.ylabel('weight')
plt.legend(loc = 'upper left')

plt.show()

plt.scatter(y_train_pred, y_train_pred - Y_train, c = 'blue',marker ='o', label = 'Train Data')
plt.scatter(y_test_pred, y_test_pred - Y_test, c = 'green', marker = 's', label ='Test Data')
plt.xlabel('предсказанное значение')
plt.ylabel('остатки')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 50, xmax = 100 , lw =2, color = 'green')
plt.xlim([50,100])
plt.show()


# %% [markdown]
# **Problem solution -2**

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
test = pd.read_csv("../input/combined-cycle-power-plant-data-set/test.csv")
train = pd.read_csv("../input/simple-linear-regression/powerplant.csv")
train = train.fillna(train.median(axis=0), axis=0)
train = (train  - train.mean()) / train.std()
print(train.tail())
print()
print(train.corr())

sns.set(style = 'whitegrid', context = 'notebook')
cols = ['AT', 'V', 'AP', 'RH', 'PE']
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale = 1)
hm = sns.heatmap(cm, cbar = True, annot=True,square = True, fmt ='.2f', annot_kws ={'size':7 },yticklabels = cols, xticklabels = cols)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

X = train.loc[:,['AT', 'V', 'AP', 'RH']].values
Y = train.PE.values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

slr = LinearRegression()
slr = slr.fit(X_train, Y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print()
print('R2 training: %.3f, test: %.3f' %(r2_score(Y_train, y_train_pred),r2_score(Y_test,y_test_pred)))
print('Mean_squared: %.3f, test: %.3f' %(mean_squared_error(Y_train, y_train_pred),mean_squared_error(Y_test,y_test_pred)))

X = train.iloc[:, :-1]
Y = train.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

slr = RandomForestRegressor(n_estimators = 1000,criterion = 'mse',random_state = 1, n_jobs = -1)
slr = slr.fit(X_train, Y_train)

y_train_pred = slr.predict(X_train)

y_test_pred = slr.predict(X_test)
print('R2 training: %.3f, test: %.3f' %(r2_score(Y_train, y_train_pred),r2_score(Y_test,y_test_pred)))
print('Mean_squared: %.3f, test: %.3f' %(mean_squared_error(Y_train, y_train_pred),mean_squared_error(Y_test,y_test_pred)))

# %% [code]
x_train_AT = train.AT.values
x_train_AT = np.resize(x_train_AT, (9567,1))
y_train_pred_graph = np.resize(y_train_pred,(9567,1))
y_test_pred_graph = np.resize(y_test_pred,(9567,1))
plt.scatter(x_train_AT, y_train_pred_graph, c = 'blue',marker ='o', label = 'Train Data')
plt.scatter(x_train_AT,y_test_pred_graph, c = 'green', marker = 's', label ='Test Data')
plt.xlabel('AT')
plt.ylabel('PE')
plt.legend(loc = 'upper left')
plt.xlim(-2,-1)
plt.ylim(-2,-1)

plt.show()
print(y_train_pred - Y_train)

plt.scatter(y_train_pred, y_train_pred - Y_train, c = 'blue',marker ='o', label = 'Train Data')
plt.scatter(y_test_pred, y_test_pred - Y_test, c = 'green', marker = 's', label ='Test Data')
plt.xlabel('предсказанное значение')
plt.ylabel('остатки')
plt.legend(loc = 'upper left')
plt.xlim(-1,1)
plt.ylim(-0.1,0.1)
plt.show()

# %% [markdown]
# **Problem solution -3**

# %% [code]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
housing = pd.read_csv("../input/california-housing-prices/housing.csv")
housing.columns = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value','ocean_proximity']
housing.ocean_proximity = pd.get_dummies(housing['ocean_proximity'])

display(housing)

housing = housing.fillna(housing.median(axis=0), axis=0)

housing = (housing  - housing.mean()) / housing.std()
print(housing.describe())
print(housing.corr())
sns.set(style = 'whitegrid', context = 'notebook')
cols = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value','ocean_proximity']
cm = np.corrcoef(housing[cols].values.T)
sns.set(font_scale = 1)
hm = sns.heatmap(cm, cbar = True, annot=True,square = True, fmt ='.2f', annot_kws ={'size':7 },yticklabels = cols, xticklabels = cols)

# %% [code]
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

X = housing.loc[:,['longitude','latitude','housing_median_age','total_rooms','ocean_proximity','median_income']].values
Y = housing['median_house_value'].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 0)

slr = RandomForestRegressor(n_estimators = 1000,criterion = 'mse',random_state = 1, n_jobs = -1)
slr = slr.fit(X_train, Y_train)

y_train_pred = slr.predict(X_train)

y_test_pred = slr.predict(X_test)
print('R2 training: %.3f, test: %.3f' %(r2_score(Y_train, y_train_pred),r2_score(Y_test,y_test_pred)))
print('Mean_squared: %.3f, test: %.3f' %(mean_squared_error(Y_train, y_train_pred),mean_squared_error(Y_test,y_test_pred)))

# %% [code]
plt.scatter(y_train_pred,y_train_pred - Y_train,c = 'green',marker = 'o', s = 35, alpha = 0.5, label = 'train data')
plt.scatter(y_test_pred,y_test_pred - Y_test, c = 'blue', marker = 's', s = 35,alpha = 0.5 , label = 'test data')

plt.show()

x_ocean_inc = np.resize(housing.ocean_proximity.values,(100,1))
y_housing_med_inc = np.resize(housing['median_house_value'].values,(100,1))

y_test_pred_graph = np.resize(y_test_pred,(100,1))
x = np.resize(np.arange(1,100),(100,1))
plt.scatter(x,y_test_pred_graph, c = 'green', marker = 's', label ='Test_Data')
plt.scatter(x,y_housing_med_inc, c = 'red', marker = 'o', label = 'Table_Data')
plt.xlabel('ocean_proximuty')
plt.ylabel('median_value')

plt.legend(loc = 'upper left')


plt.show()

plt.scatter(y_train_pred, y_train_pred - Y_train, c = 'blue',marker ='o', label = 'Train Data')
plt.scatter(y_test_pred, y_test_pred - Y_test, c = 'green', marker = 's', label ='Test Data')
plt.xlabel('предсказанное значение')
plt.ylabel('остатки')
plt.legend(loc = 'upper left')

plt.xlim(-0.1,0.1)
plt.ylim(-0.5,0.5)
plt.show()