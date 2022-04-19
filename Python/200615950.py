import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#change to the directory where the files are stored
os.chdir("/Users/yimingtang/Downloads/ST2195 Assignment/Coursework")

try:
    os.remove('airline.db')
except OSError:
    pass

#create database
conn = sqlite3.connect('airline.db')

#reading csv and writing into database
airports = pd.read_csv("airports.csv")
carriers = pd.read_csv("carriers.csv")
planes = pd.read_csv("plane-data.csv")

airports.to_sql('airports',con=conn, index=False)
carriers.to_sql('carriers',con=conn, index=False)
planes.to_sql('planes',con=conn, index=False)

c = conn.cursor()

#creating table in database for ontime
c.execute(''' 
CREATE TABLE ontime (
    Year int,
    Month int,
    DayofMonth int,
    DayofWeek int,
    DepTime int,
    CRSDepTime int,
    ArrTime int,
    CRSArrTime int,
    UniqueCarrier varchar(5),
    FlightNum int,
    TailNum varchar(8),
    ActualElapsedTime int,
    CRSElapsedTime int,
    AirTime int,
    ArrDelay int,
    DepDelay int,
    Origin varchar(3),
    Dest varchar(3),
    Distance int,
    TaxiIn int,
    TaxiOut int,
    Cancelled int,
    CancellationCode varchar(1),
    Diverted varchar(1),
    CarrierDelay int,
    WeatherDelay int,
    NASDelay int,
    SecurityDelay int,
    LateAircraftDelay int,
    TotalDelay int,
    Delay varchar(1)
    )
''')

conn.commit()


#ontime
ontime = pd.concat(map(pd.read_csv,['2005.csv','2006.csv']))
ontime=ontime.dropna(axis=0, thresh = 26)
ontime["TotalDelay"]=ontime["ArrDelay"] + ontime["DepDelay"]
ontime["Delay"]=np.where(ontime["TotalDelay"]>15, 1,0)

ontime.to_sql('ontime', if_exists = 'append',con=conn,index=False)


#=====Answer to Question 1=====
q1a=c.execute('''
SELECT ontime.Deptime, AVG(ontime.TotalDelay) as AvgDelay
FROM ontime
WHERE ontime.Cancelled = 0 AND ontime.Diverted = 0
GROUP by ontime.Deptime
ORDER by AvgDelay
''').fetchall()

q1a=pd.DataFrame(q1a,columns=['Deptime','AvgDelay'])

print(q1a['Deptime'].iloc[0], "is the best time of day to fly to minimise delays, having ",q1a['AvgDelay'].iloc[0],"average delay in minutes")
      
q1b=c.execute('''
SELECT ontime.DayofWeek, AVG(ontime.TotalDelay) as AvgDelay
FROM ontime
WHERE ontime.Cancelled = 0 AND ontime.Diverted = 0
GROUP by ontime.DayofWeek
ORDER by AvgDelay
''').fetchall()

q1b=pd.DataFrame(q1b,columns=['DayofWeek','AvgDelay'])

print(q1b['DayofWeek'].iloc[0], "is the best day of the week to fly to minimise delays, having ",q1b['AvgDelay'].iloc[0],"average delay in minutes")


q1c=c.execute('''
SELECT ontime.Month, AVG(ontime.TotalDelay) as AvgDelay
FROM ontime
WHERE ontime.Cancelled = 0 AND ontime.Diverted = 0
GROUP by ontime.Month
ORDER by AvgDelay
''').fetchall()

q1c=pd.DataFrame(q1c,columns=['Month','AvgDelay'])

print(q1c['Month'].iloc[0], "is the best time of year to fly to minimise delays, having ",q1c['AvgDelay'].iloc[0],"average delay in minutes")


#=====Answer to Question 2=====
q2=c.execute('''
SELECT planes.year As PlaneYear,ontime.Year As FlightYear, ontime.TotalDelay As TotalDelay, ontime.Delay As Delay
  FROM planes JOIN ontime USING(tailnum)
  WHERE ontime.Cancelled=0 AND planes.year !='' AND planes.year != '0000' AND planes.year !='None'
  ORDER by FlightYear DESC       
''').fetchmany(6)

q2=pd.DataFrame(q2,columns=['PlaneYear','FlightYear','TotalDelay','Delay'])

print(q2)

#Checking datatype of q2
dataTypeSeries = q2.dtypes
print(dataTypeSeries)

q2['PlaneYear']=q2['PlaneYear'].astype(int)

q2["YearDiff"] = q2["FlightYear"] - q2["PlaneYear"]
q2["Delay"]=np.where(q2["TotalDelay"] >15, "Yes","No" )

#plotting of q2

q2plot = q2.groupby(['YearDiff','Delay']).size().unstack()
q2plot.plot(kind='bar', stacked = True, ylabel = "No. of Delay", xlabel = "Year Difference")

q2bplot=q2plot.apply(lambda x: x*100/sum(x), axis=1)
q2bplot.plot(kind="bar", stacked=True)

#======Answer to Question 3=====

q3=c.execute('''
SELECT airports.state AS State, COUNT(*) AS Count, ontime.Month, ontime.Year
FROM airports JOIN ontime ON ontime.origin = airports.iata
WHERE ontime.Cancelled = 0 AND ontime.Diverted = 0 AND State !=''
GROUP by State,ontime.Year,ontime.Month
ORDER by State
''').fetchall()

q3=pd.DataFrame(q3,columns=['State','Count','Month','Year'])

print(q3)

#plotting of q3

def do_annotate(x_col, y_col, data, color, **kwargs):
    x = data[x_col]
    y = data[y_col]
    for i in range(len(x)):
        plt.annotate(str(y.values[i]), xy=(x.values[i]-1, y.values[i]), fontsize=6,
                     xytext=(0, 10), textcoords="offset points",
                     color=kwargs.get("text_color", "k"),
                     va='center', ha='center', weight='bold')
        
#As DC doesn't have flight in 2005      
q3 = q3[~((q3['State'] == 'DC') & (q3['Year'] == 2005))]

g3 = sns.catplot(kind='point', data=q3, x="Month", y="Count", hue="Year", palette='spring',
                col="State", col_wrap=8, height=2.6, aspect=1.7, sharey=False, sharex=True, legend_out=True)
g3.map_dataframe(do_annotate, 'Month', 'Count', text_color='navy')

#=====Answer to Question 4=====

#sample of tailnum derived from R for answering
q4sampletailnum={'TailNum':["N351UA","N960DL","N524", "N14998", "N355CA", "N587AA", "N839UA","N516UA"]}

q4sampletailnum=pd.DataFrame(q4sampletailnum)

q4a=c.execute('''
SELECT airports.airport AS AirportOrigin,ontime.Year, ontime.Month,ontime.DayofMonth, ontime.DepTime,ontime.ArrTime,ontime.DepDelay,ontime.ArrDelay,ontime.TotalDelay,ontime.TailNum, ontime.Origin, ontime.Dest, airports.long AS OriginLong, airports.lat AS OriginLat
FROM airports JOIN ontime ON ontime.origin = airports.iata
WHERE ontime.Year=2005 AND ontime.Month=1 AND ontime.DayofMonth BETWEEN 1 AND 7 AND ontime.Cancelled = 0 AND ontime.Diverted = 0 AND ontime.TailNum IN ('N351UA','N960DL','N524', 'N14998', 'N355CA', 'N587AA', 'N839UA','N516UA')
ORDER by ontime.TailNum, ontime.Year,ontime.Month,ontime.DayofMonth, ontime.DepTime,ontime.ArrTime
''').fetchall()

q4a=pd.DataFrame(q4a,columns=['AirportOrigin','Year','Month','DayofMonth','Deptime','ArrTime','DepDelay','ArrDelay','TotalDelay','TailNum','Origin','Dest','OriginLong','OriginLat'])

print(q4a)

q4b=c.execute('''
SELECT airports.airport AS AirportDest,ontime.Year, ontime.Month,ontime.DayofMonth, ontime.DepTime,ontime.ArrTime,ontime.DepDelay,ontime.ArrDelay,ontime.TotalDelay,ontime.TailNum, ontime.Origin, ontime.Dest, airports.long AS DestLong, airports.lat AS DestLat
FROM airports JOIN ontime ON ontime.dest = airports.iata
WHERE ontime.Year=2005 AND ontime.Month=1 AND ontime.DayofMonth BETWEEN 1 AND 7 AND ontime.Cancelled = 0 AND ontime.Diverted = 0 AND ontime.TailNum IN ('N351UA','N960DL','N524', 'N14998', 'N355CA', 'N587AA', 'N839UA','N516UA')
ORDER by ontime.TailNum, ontime.Year,ontime.Month,ontime.DayofMonth, ontime.DepTime,ontime.ArrTime
''').fetchall()

q4b=pd.DataFrame(q4b,columns=['AirportDest','Year','Month','DayofMonth','Deptime','ArrTime','DepDelay','ArrDelay','TotalDelay','TailNum','Origin','Dest','DestLong','DestLat'])

print(q4b)

q4c=pd.merge(q4a, q4b, how="inner" ,on=['Year','Month','DayofMonth','Deptime','ArrTime','DepDelay','ArrDelay','TotalDelay','TailNum','Origin','Dest'])

print(q4c)

#plotting of q4

#basemap for plotting
import geopandas as gpd

states = gpd.read_file("/Users/yimingtang/Downloads/ST2195 Assignment/Coursework/tl_2021_us_state/tl_2021_us_state.shp")
states= states.to_crs("EPSG:4326")

non_continental = ['HI','VI','MP','GU','AK','AS','PR']


for n in non_continental:
    states = states[states.STUSPS != n]

#q4a(1)

q4c1 = q4c[(q4c.TailNum=='N351UA')]

#zip disappear after creating a object
source_to_dest1 = zip(q4c1["OriginLat"], q4c1["DestLat"],
                     q4c1["OriginLong"], q4c1["DestLong"],
                     q4c1["TotalDelay"],(q4c1["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest1:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='red', alpha=0.5)
    plt.title("N351UA")
 
#q4a(2)

q4c2 = q4c[(q4c.TailNum=='N960DL')]

#zip disappear after creating a object
source_to_dest2 = zip(q4c2["OriginLat"], q4c2["DestLat"],
                     q4c2["OriginLong"], q4c2["DestLong"],
                     q4c2["TotalDelay"],(q4c2["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest2:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='orangered', alpha=0.5)
        
    plt.title("N960DL")
    
#q4a(3)

q4c3 = q4c[(q4c.TailNum=="N524")]

#zip disappear after creating a object
source_to_dest3 = zip(q4c3["OriginLat"], q4c3["DestLat"],
                     q4c3["OriginLong"], q4c3["DestLong"],
                     q4c3["TotalDelay"],(q4c3["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest3:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='orange', alpha=0.5)
        
    plt.title("N524")

#q4a(4)

q4c4 = q4c[(q4c.TailNum=="N14998")]

#zip disappear after creating a object
source_to_dest4 = zip(q4c4["OriginLat"], q4c4["DestLat"],
                     q4c4["OriginLong"], q4c4["DestLong"],
                     q4c4["TotalDelay"],(q4c4["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest4:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='gold', alpha=0.5)
        
    plt.title("N14998")     

#q4a(5)

q4c5 = q4c[(q4c.TailNum=="N355CA")]

#zip disappear after creating a object
source_to_dest5 = zip(q4c5["OriginLat"], q4c5["DestLat"],
                     q4c5["OriginLong"], q4c5["DestLong"],
                     q4c5["TotalDelay"],(q4c5["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest5:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='chartreuse', alpha=0.5)
        
    plt.title("N355CA") 

#q4a(6)

q4c6 = q4c[(q4c.TailNum=="N587AA")]

#zip disappear after creating a object
source_to_dest6 = zip(q4c6["OriginLat"], q4c6["DestLat"],
                     q4c6["OriginLong"], q4c6["DestLong"],
                     q4c6["TotalDelay"],(q4c6["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest6:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='deepskyblue', alpha=0.5)
        
    plt.title("N587AA")    

#q4a(7)

q4c7 = q4c[(q4c.TailNum=="N839UA")]

#zip disappear after creating a object
source_to_dest7 = zip(q4c7["OriginLat"], q4c7["DestLat"],
                     q4c7["OriginLong"], q4c7["DestLong"],
                     q4c7["TotalDelay"],(q4c7["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest7:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='blue', alpha=0.5)
        
    plt.title("N839UA")  

#q4a(8)

q4c8 = q4c[(q4c.TailNum=="N516UA")]

#zip disappear after creating a object
source_to_dest8 = zip(q4c8["OriginLat"], q4c8["DestLat"],
                     q4c8["OriginLong"], q4c8["DestLong"],
                     q4c8["TotalDelay"],(q4c8["TailNum"]))


with plt.style.context(("seaborn", "ggplot")):
    ## Plot world
    states.plot(figsize=(15,15), edgecolor="grey", color="white");

    ## Loop through each flight plotting line depicting flight between source and destination
    ## We are also plotting scatter points depicting source and destinations
    for slat,dlat,slon,dlon, TotalDelay,TailNum in source_to_dest8:
        plt.plot([slon , dlon], [slat, dlat], linewidth=TotalDelay/75, color='purple', alpha=0.5)
        
    plt.title("N516UA")             
    
#plotting of q4b

q4c["id"] = q4c.index + 1 

def do_annotate2(x_col, y_col, data, color, **kwargs):
    x = data[x_col]
    y = data[y_col]
    z = data["AirportOrigin"]
    for i in range(len(x)):
        plt.annotate(str(z.values[i]), xy=(x.values[i]-1, y.values[i]), fontsize=3,
                     xytext=(0, 10), textcoords="offset points",
                     color=kwargs.get("text_color", "k"),
                     va='center', ha='center', weight='bold')

g4 = sns.catplot(kind='point', data=q4c, x="id", y="TotalDelay", hue="TailNum", palette='spring',
                col="TailNum", col_wrap=4, height=5, aspect=1.7, sharey=False, sharex=False, legend_out=True)
g4.map_dataframe(do_annotate2, 'id', 'TotalDelay', text_color='navy')
plt.show()

    
#=====Answer to Question 5=====
q5new = ontime.sample(n=5000,random_state=1) #creating random subset of n=5000 from ontime

q5new.to_csv('q5new.csv', index=None) #exporting to r as python sample can be replicated
 
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV      
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #transform different types


features = ['Year', 'Month', 'DayofMonth', 'DepTime','TailNum']
X = q5new[features].copy()
y= q5new["Delay"]

numerical_features = ['Year', 'Month', 'DayofMonth','DepTime']

# Applying SimpleImputer and StandardScaler into a pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())])

categorical_features = ['TailNum']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

data_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)]) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

param_grid = {
    'data_transformer__numerical__imputer__strategy': ['mean', 'median'],
    'data_transformer__categorical__imputer__strategy': ['constant','most_frequent']
}

#Logistic Regression
pipe_lr = Pipeline(steps=[('data_transformer', data_transformer),
                      ('pipe_lr', LogisticRegression(max_iter=10000, penalty = 'none'))])
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid)
grid_lr.fit(X_train, y_train);

#Penalised Logistic Regression
pipe_plr = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_plr', LogisticRegression(penalty='l1', max_iter=10000, tol=0.01, solver='saga'))])
grid_plr = GridSearchCV(pipe_plr, param_grid=param_grid)
grid_plr.fit(X_train, y_train);

#Gradient Boosting
pipe_gdb = Pipeline(steps=[('data_transformer', data_transformer),
       ('pipe_gdb',GradientBoostingClassifier(random_state=2))])

grid_gdb = GridSearchCV(pipe_gdb, param_grid=param_grid)
grid_gdb.fit(X_train, y_train);

#Classification Tree
pipe_tree = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_tree', DecisionTreeClassifier(random_state=0))])
grid_tree = GridSearchCV(pipe_tree, param_grid=param_grid)
grid_tree.fit(X_train, y_train);

#Random Forest
pipe_rf = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_rf', RandomForestClassifier(random_state=0))])
grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid)
grid_rf.fit(X_train, y_train);

#Support Vector Machine
pipe_svm = Pipeline(steps=[('data_transformer', data_transformer),
                           ('pipe_svm',  LinearSVC(random_state=0, max_iter=10000, tol=0.01))])
grid_svm = GridSearchCV(pipe_svm, param_grid=param_grid)
grid_svm.fit(X_train, y_train);

#Compare ROC Curve

ax = plt.gca()
plot_roc_curve(grid_lr, X_test, y_test, ax=ax, name='Logistic Regression')
plot_roc_curve(grid_gdb, X_test, y_test, ax=ax, name='Gradient Boosting')
plot_roc_curve(grid_plr, X_test, y_test, ax=ax, name='Penalised logistic regression')
plot_roc_curve(grid_tree, X_test, y_test, ax=ax, name='Classification trees')
plot_roc_curve(grid_rf, X_test, y_test, ax=ax, name='Random forests')
plot_roc_curve(grid_svm, X_test, y_test, ax=ax, name='Support vector machines')
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.show()

#============================

