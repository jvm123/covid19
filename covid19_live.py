#!/usr/bin/env python
# coding: utf-8

# In[1]:


# strongly inspired by https://towardsdatascience.com/analyzing-coronavirus-covid-19-data-using-pandas-and-plotly-2e34fe2c4edc
# download population data from https://data.worldbank.org/indicator/sp.pop.totl as csv, unzip and manually remove the first four lines and replace 'United States' with 'US' and 'Iran, Islamic Rep.' with 'Iran', save to pop.csv
## Import Libraries
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
plt.rcParams['figure.figsize'] = [15, 5]
from IPython import display
from ipywidgets import interact, widgets
import datetime

## Read Data for Cases, Deaths and Recoveries
srcconfirmed="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
srcrecovered="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
srcdeaths="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
srcpop="pop.csv"
Cases_confirmed=pd.read_csv(srcconfirmed)
Cases_recovered=pd.read_csv(srcrecovered)
Cases_deaths=pd.read_csv(srcdeaths)
pop_raw=pd.read_csv(srcpop)

print("Data source: https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/")
print("Retrieved on: " + str(datetime.datetime.now()))


# In[2]:


pop = pd.DataFrame([pop_raw['2018'].values], columns=pop_raw['Country Name'].values)
pop['Iran'].values[0]


# In[3]:


def fix_date(df):
    return datetime.datetime.strptime(df["date"], '%m/%d/%y').strftime('%Y-%m-%d')

def fix_dataset(cases, value_name):
    cases = cases.rename(columns={'Province/State':'label', 'Country/Region':'parent'})
    cases.drop(['Lat', 'Long'], axis=1, inplace=True)
    cases = cases.melt(id_vars=['label','parent'], var_name="date", value_name=value_name)
    cases.fillna(value="", inplace=True)
    cases["date"] = cases.apply(fix_date, axis=1)
    cases.set_index(["parent","label","date"], inplace=True)
    return cases

# Transform
Cases_confirmed_clean = fix_dataset(Cases_confirmed, "confirmed")
Cases_recovered_clean = fix_dataset(Cases_recovered, "recovered")
Cases_deaths_clean = fix_dataset(Cases_deaths, "deaths")

Cases_raw = Cases_deaths_clean.merge(Cases_confirmed_clean, left_on=['parent','label','date'], right_on=['parent','label','date'])
Cases_raw = Cases_raw.merge(Cases_recovered_clean, left_on=['parent','label','date'], right_on=['parent','label','date'])
Cases_raw.loc['US']


# In[4]:


def add_derived(Cases):
#    Cases.set_index('date', inplace=True)
#    Cases['active'] = Cases['confirmed']-Cases['deaths']-Cases['recovered']
    Cases['lethality'] = np.round(Cases['deaths']/Cases['confirmed'], 3)
    
#    Cases.drop(['derivedConfirmed_'], axis=1, inplace=True)
    return Cases

# Get Daily Data
#Cases_diff = Cases_raw.groupby(['parent','label','date'])
#Cases_diff = Cases_diff.sum()
#Cases_diff = Cases_diff.diff().fillna(0)
#Cases_diff = Cases_diff.rename(columns={"confirmed":"confirmed_new","deaths":"deaths_new","recovered":"recovered_new"})
#Cases = Cases_raw.merge(Cases_diff, left_on=['parent','label','date'], right_on=['parent','label','date'])

Cases = Cases_raw
Cases.loc['Germany']
#Cases


# In[5]:


# Shift data
#CasesShiftedB = Cases.tail(0)
#for country in Cases.droplevel('date').index.unique().tolist():
#    firstcase = Cases.loc[country]['deaths'].reset_index().set_index('date')
#    firstcase = firstcase[firstcase.ne(0)].dropna().reset_index()
#    firstcase['parent'] = country
#    firstcase = firstcase.set_index(['parent', 'label', 'date'])
#    CasesShiftedB = CasesShiftedB.append(firstcase)
#CasesShiftedB

#CasesS = {}

#for place in Cases.droplevel('date').index.unique().tolist():
#    firstcase = Cases.loc[place]['deaths'].reset_index().set_index('date')
#    firstcase = firstcase[firstcase.ne(0)]
#    CasesS[place] = firstcase

#CasesS

def add_day(df):
    firstcase = Cases.loc[place]['deaths'].reset_index().set_index('date')
    Cases['day'] = Cases.loc[place][firstcase.ne(0)]
    return day

def combine_projectedDeaths(row):
    if row['deaths'] == False:
        return row['projectedDeaths']
    return row['deaths']

def add_simulated(Cases, place):
    Cases['deaths_bak'] = Cases['deaths']
    Cases['deaths'] = Cases.apply(combine_projectedDeaths, axis=1)
    Cases['derivedConfirmed_'] = np.round(Cases['deaths'] / 0.03, 0) # mortality
    Cases['derivedConfirmed'] = Cases['derivedConfirmed_'].shift(periods=-14) # time to death after turning infectious
    
#    Cases['derivedRecovered_'] = Cases['derivedConfirmed_'].shift(periods=-3) # periods = time to death + time to recovery
#    Cases['derivedRecovered'] = Cases['derivedRecovered_'] - Cases['deaths']
    
#    Cases['derivedActive'] = Cases['derivedConfirmed_'].shift(periods=-12) - Cases['deaths'] - Cases['derivedRecovered'] # periods = time to death + incubation period
    Cases['derivedLethality'] = np.round(Cases['deaths']/Cases['derivedConfirmed'], 3)
    Cases['testCoverage'] = np.minimum(1,np.round(Cases['confirmed']/Cases['derivedConfirmed'], 3))
    
    per1mio = ["deaths", "confirmed", "recovered", "projectedDeaths", "derivedConfirmed"]
    for col in per1mio:
        Cases[col+"_per1mio"] = np.round(Cases[col] * (1000000 / pop[place].values[0]), 0)
    
    Cases['derivedConfirmed'] = np.round(Cases['derivedConfirmed'], 0)
    Cases['deaths'] = Cases['deaths_bak']
    #Cases.drop(['derivedConfirmed_','derivedRecovered_','deaths_bak'], axis=1, inplace=True)
    Cases.drop(['derivedConfirmed_','deaths_bak'], axis=1, inplace=True)
    return Cases

#Cases_sim = add_simulated(Cases.loc['Germany'].droplevel('label'))

# parameters and dates
lookahead = 90
cutoffsimulated = -15 # days we calculate in the simulation, but later don't display
cutoffpolynomial = -14 # days we calculate in the polynomial, but later don't display

firstrecordeddate = Cases.reset_index().head(1)['date'].values[0]
firstsimdate = Cases.reset_index().head(2)['date'].values[-1]
lastrecordeddate = Cases.reset_index().tail(1)['date'].values[0]
startdate = datetime.datetime.strptime(lastrecordeddate, '%Y-%m-%d')
lastrecordeddatereadable = startdate.strftime('%d. %B %Y')
projectionstartdate = (startdate + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
#startdate = datetime.datetime.strptime(cases.index[-1], '%Y-%m-%d')
enddate = (startdate + datetime.timedelta(days=(lookahead + cutoffsimulated)))
enddatereadable = enddate.strftime('%d. %B %Y')
enddate = enddate.strftime('%Y-%m-%d')
datetoday = datetime.datetime.now().strftime('%Y-%m-%d')
datereadable = datetime.datetime.now().strftime('%d. %B %Y')

def extenddata(cases, place):
    cases = cases.reset_index().groupby(["date"]).sum() #.reset_index().set_index('date')
    y = cases['deaths'].values

    # average every two entries
    y = y[-13:]
    #y = np.nanmean(np.pad(y.astype(float), (0, 3 - y.size%3), mode='constant', constant_values=np.NaN).reshape(-1, 3), axis=1)
    
    n = np.array(range(1,14))
    n = n*1/14
    
    # calculate polynomial
    x = range(0, y.size)
    z = np.polyfit(x, y, 2)#, w=n)#np.sqrt(n))
    f = np.poly1d(z)
    print(f)

    # calculate new x's and y's
    x_new = np.linspace(x[-1], x[-1]+lookahead, lookahead)
    y_new = f(x_new)
    
    # prepare new date range
    x_new_dates = [] # in the fit graph we need to start one day early for it to look smooth
    last_val = cases['deaths'].values[-1]
    for x_delta in range(1,lookahead): # generate future dates
        date = (startdate + datetime.timedelta(days=x_delta)).strftime('%Y-%m-%d')
        x_new_dates.append(date)
        val = max(0,np.round(f(x[-1]+x_delta)),0)
        
        # once we are over the peak deaths, the curve ends and goes linear
        if val < last_val:
            val = last_val
        last_val = val
        
        cases = cases.append(pd.DataFrame({'date':[date], 'deaths': False, 'projectedDeaths':[val]}).set_index("date"))
#        print('delta '+date+" y "+str(f(x[-1]+1+x_delta)))

    # cut displayed data and add simulated numbers
#    date = (startdate + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
#    cases = add_simulated(cases)[date:enddate].fillna(0)
    cases = add_derived(cases)
    cases = add_simulated(cases, place)[firstsimdate:enddate].fillna(0)
#    y_new = y_new[:cutoffpolynomial]

    # export data
    title = place
    if title != "":
        cases.to_csv("datasets/"+datetoday+"_sim_"+title+".csv")
        cases.to_html("datasets/"+datetoday+"_sim_"+title+".html")
#        cases.to_excel("output.xslx")

    # in the fit graph we need to start one day early for it to look smooth
    #cases = cases.append(pd.DataFrame({'date':[startdate], 'deaths':[f(x[-1]+1)]}).set_index("date"))
        
    return cases

GlobalTotals = Cases.reset_index().groupby('date').sum()
#GlobalTotals = extenddata(GlobalTotals)
#GlobalTotals.loc["2020-03-01":"2020-03-14"]
#GlobalTotals
#Cases.loc['US']
extenddata(Cases.loc['China'], 'China')
#extenddata(Cases.loc['Germany'], 'Germany').loc[lastrecordeddate]
#extenddata(Cases.loc['Germany'], 'Germany').loc[projectionstartdate]
#Cases.loc['Germany']


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-dark')

Cases = Cases.reset_index().set_index(['parent','label'])

def plotData(cases, place, show=True, sim=False):
    title = place
    cases = cases.groupby('date').sum().reset_index().set_index('date')
    if type(sim) == bool and sim == False:
        sim = extenddata(cases, place)
    
#    sim = extenddata(cases[firstrecordeddate:"2020-04-04"], title)
    projected = sim[lastrecordeddate:]
    
    fig = make_subplots(rows=3, cols=2,shared_xaxes=True,
                        specs=[[{}, {}],[{},{}],
                           [{"colspan": 2}, None]],
                        subplot_titles=('Total Confirmed Cases','Active Cases','Deaths','Recoveries','Death to Cases Ratio'))
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=cases.index,y=cases['confirmed'],
                             mode='lines',
                             name='Confirmed Cases',
                             line=dict(color='red',width=2)))
#                             row=1,col=1)
    #fig.add_trace(go.Scatter(x=cases.index,y=cases['active'],
    #                         mode='lines',
    #                         name='Active Cases',
    #                         line=dict(color='purple',width=2)))
#                             row=1,col=2)
    #fig.add_trace(go.Scatter(x=cases.index,y=cases['recovered'],
    #                         mode='lines',
    #                         name='Recoveries',
    #                         line=dict(color='green',width=2)))
#                             row=1,col=1)
    fig.add_trace(go.Scatter(x=cases.index,y=cases['deaths'],
                             mode='lines',
                             name='Deaths',
                             line=dict(color='black',width=2)))
#                             row=1,col=1)
    fig.add_trace(go.Scatter(x=projected.index,y=projected['projectedDeaths'],
                             mode='lines',
                             name='Projected deaths',
                             line=dict(color='black',width=2,dash='dot')))
#                             row=1,col=1)
        
    fig.add_trace(go.Scatter(x=sim.index,y=sim['derivedConfirmed'],
                             mode='lines',
                             name='Derived Cases',
                             line=dict(color='red',width=2,dash='dot')))
                             #row=1,col=1)
    #fig.add_trace(go.Scatter(x=sim.index,y=sim['derivedActive'],
    #                         mode='lines',
    #                         name='Derived Active Cases',
    #                         line=dict(color='purple',width=2,dash='dot')))
                             #row=1,col=2)
    #fig.add_trace(go.Scatter(x=sim.index,y=sim['derivedRecovered'],
    #                         mode='lines',
    #                         name='Derived Recoveries',
    #                         line=dict(color='green',width=2,dash='dot')))
                             #row=2,col=2)

    fig.update_layout(showlegend=True)
#    fig.update_layout(title=title,
#                       yaxis_title='Cases',
#                       xaxis_title='Date')
#    fig.show()
    fig.update_layout(title=title,#+" (Projection from "+datereadable+")",
#                       yaxis_title='Cases',
                       yaxis_title='Cases (log)', yaxis_type="log",
                       xaxis_title='Date')

    if show:
        fig.show()
    fig.write_html("datasets/"+datetoday+"_cases_"+title+".html")
#    fig.write_image("test.png")

#plotData(GlobalTotals, 'Global')
plotData(Cases.loc["US"], 'US')
#plotData(Cases.loc['United Kingdom'], 'United Kingdom')
#plotData(Cases.loc['China'], 'China')
#plotData(Cases.loc['Italy'], 'Italy')
#plotData(Cases.loc['Germany'], 'Germany')
#plotData(Cases.loc['Sweden'], 'Sweden')
#plotData(Cases.loc['Belgium'], 'Belgium')
#plotData(Cases.loc['France'], 'France')
#plotData(Cases.loc['Spain'], 'Spain')
#plotData(Cases.loc['Netherlands'], 'Netherlands')


# In[7]:


Cases.loc['United Kingdom']


# In[8]:


Cases.loc['Germany']


# In[9]:


countries = ["France","Spain","US","United Kingdom","China","Netherlands","Germany","Belgium","Sweden"]
sims_lastrecordeddate = {}
sims_enddate = {}
columns = {}
per1mio = ["deaths", "confirmed", "recovered", "projectedDeaths"]
for place in countries:
    print(place)
    sim = extenddata(Cases.loc[place], place)
    plotData(Cases.loc[place], place, False, sim) # export interactive plot
    
    sims_lastrecordeddate[place] = sim.loc[lastrecordeddate]
    sims_enddate[place] = sim.loc[enddate]
#    if "label" not in sims_lastrecordeddate[place].index.values:
#        sims_lastrecordeddate[place] = sims_lastrecordeddate[place].reset_index().groupby("date").sum().loc[lastrecordeddate]
#        sims_enddate[place] = sims_enddate[place].drop("label")
#    else:
#        sims_lastrecordeddate[place] = sims_lastrecordeddate[place].drop("label")
#        sims_enddate[place] = sims_enddate[place].drop("label")
    columns = sims_lastrecordeddate[place].index
    # TODO: exclude testCoverage, lethality from division
    
    #for col in per1mio:
    #    sims_lastrecordeddate[place][col] = np.round(sims_lastrecordeddate[place][col] * (1000000 / pop[place].values[0]), 0)
    #    sims_enddate[place][col] = np.round(sims_enddate[place][col]  * (1000000 / pop[place].values[0]), 0)
    print("ok")
    
pd_sims = pd.DataFrame(sims_lastrecordeddate, columns = sims_lastrecordeddate.keys())
pd_sims.insert(0, 'name', columns)
pd_sims.set_index('name', inplace=True)
pd_sims_enddate = pd.DataFrame(sims_enddate, columns = sims_enddate.keys())
pd_sims_enddate.insert(0, 'name', columns)
pd_sims_enddate.set_index('name', inplace=True)


# In[10]:


pd_sims_enddate
#pd_sims


# In[11]:


totalcases = pd_sims.loc['deaths_per1mio'].sort_values(ascending=False)
top10 = totalcases.head(10)
fig = go.Figure(go.Bar(x=top10.index, y=top10.values,
                      text=top10.values,
            textposition='outside'))
fig.update_layout(title_text='Top 10 Countries by Deaths / 1 Mio on '+lastrecordeddatereadable)
fig.update_yaxes(showticklabels=False)

fig.show()
fig.write_html("datasets/"+datetoday+"_top10_deaths.html")


# In[12]:


totalcases = pd_sims.loc['testCoverage'].sort_values(ascending=False)
top10 = totalcases.head(10)
fig = go.Figure(go.Bar(x=top10.index, y=top10.values,
                      text=top10.values,
            textposition='outside'))
fig.update_layout(title_text='Top Countries by Test Coverage on '+lastrecordeddatereadable)
fig.update_yaxes(showticklabels=False)

fig.show()
fig.write_html("datasets/"+datetoday+"_top10_testcoverage.html")


# In[13]:


totalcases = pd_sims.loc['derivedConfirmed_per1mio'].sort_values(ascending=False)
top10 = totalcases.head(10)
fig = go.Figure(go.Bar(x=top10.index, y=top10.values,
                      text=top10.values,
            textposition='outside'))
fig.update_layout(title_text='Top Countries by Derived Cases / 1 Mio on '+lastrecordeddatereadable)
fig.update_yaxes(showticklabels=False)

fig.show()
fig.write_html("datasets/"+datetoday+"_top10_derivedcases.html")


# In[14]:


totalcases = pd_sims_enddate.loc['projectedDeaths_per1mio'].sort_values(ascending=False)
top10 = totalcases.head(10)
fig = go.Figure(go.Bar(x=top10.index, y=top10.values,
                      text=top10.values,
            textposition='outside'))
fig.update_layout(title_text='Top Countries by Projected Deaths  / 1 Mio, '+datereadable+" => "+enddatereadable)
fig.update_yaxes(showticklabels=False)

fig.show()
fig.write_html("datasets/"+datetoday+"_top10_projecteddeaths.html")


# In[15]:


totalcases = pd_sims_enddate.loc['projectedDeaths_per1mio'].sort_values(ascending=False)
totalcases


# In[ ]:




