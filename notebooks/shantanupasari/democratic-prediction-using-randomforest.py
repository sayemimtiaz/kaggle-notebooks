
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#Reading Demographics CSV File
demographics=pd.read_csv('../input/county_facts.csv')
demographics = demographics[['fips','area_name','state_abbreviation','PST045214','AGE775214','RHI225214','RHI725214','RHI825214','EDU635213','EDU685213','INC110213','PVY020213','POP060210']]
demographics.rename(columns={'PST045214': 'Population', 'AGE775214': 'Age > 65','RHI225214':'Black','RHI725214':'Latino','RHI825214':'White','EDU635213':'HighSchool','EDU685213':'Bachelors','INC110213':'Median Household','PVY020213':'< Powerty level','POP060210':'Population PSM'}, inplace=True)

#Reading Results CSV File
results=pd.read_csv('../input/primary_results.csv')
results = results[results.party == "Democrat"]
results = results[(results.state != "Maine") & (results.state != "Massachusetts") & (results.state != "Vermont") & (results.state != "Illinois") ]
results = results[(results.candidate != ' Uncommitted') & (results.candidate != 'No Preference')]
results = results[(results.candidate == "Hillary Clinton") |(results.candidate == "Bernie Sanders") ]
Dem=results

#Calculating statewise total votes and fraction votes
votesByState = [[candidate, state, party] for candidate in Dem.candidate.unique() for state in Dem.state.unique() for party in Dem.party.unique()]
for i in votesByState:
	i.append(Dem[(Dem.candidate == i[0]) & (Dem.state == i[1])].votes.sum())
	i.append(i[3].astype('float')/Dem[Dem.state == i[1]].votes.sum())
	
print votesByState
#Merging demographics and results	
vbs = pd.DataFrame(votesByState, columns = ['candidate', 'state', 'party', 'votes','partyFrac'])
allData = pd.merge(vbs, demographics, how="inner", left_on = 'state',right_on = 'area_name')
allData.drop('state_abbreviation',axis=1, inplace=True)

#Segregate data candidate wise
HRC = allData[(allData.candidate == "Hillary Clinton")]
HRC=HRC.reset_index();

print HRC