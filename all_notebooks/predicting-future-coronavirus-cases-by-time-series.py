# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# # Importing Libraries

# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,10

# %% [markdown]
# # Reading Data

# %% [code]
data=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
print(data.shape)
data.head()

# %% [markdown]
# # Dropping Serial Number and Last Update columns for now as they wont be of a great value
# # Also creating a new column for Active Cases of Corona Virus

# %% [code]
#Dropping the columns Sno and Last Update
data.drop(['SNo','Last Update'],axis=1,inplace=True)

#Getting additonal Active cases for better implementation and information
data['Active']=data['Confirmed']-data['Deaths']-data['Recovered']

data.tail()

# %% [markdown]
# # Checking and Handling NA values

# %% [code]
print(data.isna().sum())

data['Province/State']=data['Province/State'].fillna('Not Specified')
print('\n\n',data.isna().sum())

# %% [markdown]
# # Changing the data types

# %% [code]
print(data.dtypes)

data['Confirmed']=data['Confirmed'].astype(int)
data['Recovered']=data['Recovered'].astype(int)
data['Deaths']=data['Deaths'].astype(int)
data['Active']=data['Active'].astype(int)

print('\n\n',data.dtypes)

# %% [markdown]
# # Only Cases in China

# %% [code]
china_cases=data[data['ObservationDate']==max(data['ObservationDate'])].reset_index(drop=True)
china_cases=china_cases.groupby('Country/Region')['Confirmed'].sum()['Mainland China']
china_cases

# %% [markdown]
# # Considering a variable consisting cases of Non-China

# %% [markdown]
# # As the cases in China will not play a significant role in prediciting, we drop those rows.
# # Values with _nc are the values with notchina

# %% [code]
data_nc=data[data['Country/Region']!='Mainland China']
data_nc

# %% [code]
#China cases included
data_per_day=data.groupby('ObservationDate')[['Confirmed','Deaths','Recovered','Active']].sum()
#China cases excluded
data_per_day_nc=data_nc.groupby('ObservationDate')[['Confirmed','Deaths','Recovered','Active']].sum()

# %% [markdown]
# # Plot for No.of cases including China

# %% [code]
data_per_day.plot(kind='line',figsize=(20,8))
plt.ylabel('Number of Cases',size=20)
plt.xlabel('Dates',size=20)
plt.title('Number of cases including China(Initially)',size=20)
plt.legend(prop={'size':'15'})

# %% [markdown]
# # Plot for No.of cases excluding China

# %% [code]
#Data for Countries except China
data_per_day_nc.plot(kind='line',figsize=(20,8))
plt.ylabel('Number of Cases',size=20)
plt.xlabel('Dates',size=20)
plt.title('Number of cases excluding China(Initially)',size=20)
plt.legend(prop={'size':'15'})

# %% [markdown]
# # We are using fbprophet library for time-series analysis

# %% [code]
from fbprophet import Prophet

p=Prophet()

# %% [code]
p.add_seasonality(name='monthly',period=30.5,fourier_order=5)

# %% [code]
print(data_per_day.shape)

cases=data_per_day.reset_index()
cases_nc=data_per_day_nc.reset_index()

# %% [code]
confirmed_cases=cases_nc[['ObservationDate','Confirmed']]
recovered_cases=cases_nc[['ObservationDate','Recovered']]
death_cases=cases_nc[['ObservationDate','Deaths']]
active_cases=cases_nc[['ObservationDate','Active']]

# %% [code]
confirmed_cases.rename(columns={'ObservationDate':'ds','Confirmed':'y'},inplace=True)

# %% [code]
#Fit Model
p.fit(confirmed_cases)

# %% [code]
#Future Dates
future_dates=p.make_future_dataframe(periods=30)
future_dates

# %% [code]
#Prediction
prediction=p.predict(future_dates)

# %% [markdown]
# # Plot for Predicted values of confirmed cases

# %% [code]
#Plot Prediction
p.plot(prediction,figsize=(20,8))
plt.xlabel('Dates',size=20)
plt.ylabel('Number of Confirmed cases',size=20)
plt.title('Predicted Number of Confirmed Cases',size=20)

# %% [markdown]
# # Below graphs shows the trends at which the cases would rise

# %% [code]
p.plot_components(prediction)

# %% [code]
#Find Points/Dates for change
from fbprophet.plot import add_changepoints_to_plot
fig=p.plot(prediction)
c=add_changepoints_to_plot(fig.gca(),p,prediction)

# %% [code]
prediction.tail().T

# %% [markdown]
# # The yhat consists of the values we need, therefore getting those values

# %% [code]
prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# %% [markdown]
# # Predicted values

# %% [code]
k=len(prediction)
for i in range(confirmed_cases.shape[0],k) :
  print('Prediction of Confirmed cases for',prediction['ds'][i],'is ',round(prediction['yhat'][i].astype(int))+china_cases)

# %% [markdown]
# # So code might not give the accurate predictions but are close enough.

# %% [code]
data1=data[data['ObservationDate'] == max(data['ObservationDate'])].reset_index(drop=True)
df = data1.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
df1 = df.sort_values(by='Confirmed', ascending=False).reset_index(drop=True)
df1 = df1[['Country/Region', 'Confirmed', 'Active', 'Deaths', 'Recovered']].reset_index(drop=True)

# %% [code]
from IPython.display import display, HTML
display(HTML(df1.to_html()))

# %% [code]
ab=prediction[prediction['ds']>=max(data['ObservationDate'])][['ds','yhat']].reset_index(drop=True)
ab.rename(columns={'ds':'date','yhat':'confirmed_val'},inplace=True)
ab

# %% [code]
cd=pd.DataFrame(ab)
cd.to_csv('submission.csv',index=False)