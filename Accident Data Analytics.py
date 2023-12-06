import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from contextily import add_basemap
from scipy.stats import chi2_contingency
import numpy as np

# Data Loading and Cleaning
accidents = pd.read_csv("Accidents.csv").drop_duplicates()
accidents['Date'] = pd.to_datetime(accidents['Date'])
accident_cleaned = accidents.dropna()

# Mode Function
def mode(x):
    return x.mode().iloc[0]

# Police Force Analysis
I1 = accident_cleaned['Police_Force'].value_counts().reset_index()
I1.columns = ['Police_Force', 'Number_of_Accidents']
I1 = I1.sort_values(by='Number_of_Accidents', ascending=False)
top_police_forces = I1.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Police_Force', y='Number_of_Accidents', data=top_police_forces, palette='viridis')
plt.title('Accidents across different Police Forces')
plt.xlabel('Police Forces')
plt.ylabel('Accidents')
plt.show()

# Police Force and Accident Severity Analysis
I1_PF_AS = accident_cleaned.groupby(['Police_Force', 'Accident_Severity']).size().reset_index(name='count')
I1_PF_AS = I1_PF_AS.nlargest(20, 'count')

fig = px.bar(I1_PF_AS, x='Police_Force', y='count', color='Accident_Severity',
             labels={'count': 'Number of Accidents', 'Police_Force': 'Police Force'},
             title='Number of accidents classified by Severity')
fig.show()

# Day of the Week Analysis
I2 = accident_cleaned['Day_of_Week'].value_counts().reset_index()
I2.columns = ['Day_of_the_Week', 'Number_of_Accidents']

plt.figure(figsize=(10, 6))
sns.barplot(x='Number_of_Accidents', y='Day_of_the_Week', data=I2, palette='gray_r')
plt.title('Accidents across different Days of the Week')
plt.xlabel('Accidents')
plt.ylabel('Day of The Week')
plt.show()

# Time and Day/Night Analysis
I3 = accident_cleaned[['Accident_Severity', 'Time']]
I3[['Hour', 'Minutes']] = I3['Time'].str.split(':', expand=True)
I3 = I3.dropna()
I3['Time_slot'] = pd.to_numeric(I3['Hour'])
num_sev = I3['Accident_Severity'].value_counts()
hour_slot = I3['Time_slot'].value_counts()

# Classification morning and evening
I3['DayOrNight'] = np.where((I3['Time_slot'] <= 6) | (I3['Time_slot'] >= 18), 'night', 'day')
contingency_table = pd.crosstab(I3['DayOrNight'], I3['Accident_Severity'])
chi2, _, _, _ = chi2_contingency(contingency_table)
print(f"Chi-squared test statistic: {chi2}")

# Plotting for I3
plt.figure(figsize=(10, 6))
sns.countplot(x='DayOrNight', hue='Accident_Severity', data=I3)
plt.title('Accident Severity across Day and Night')
plt.xlabel('Part of the Day')
plt.ylabel('Number of Accidents')
plt.show()

# Plotting Local Authority District
I6_LAD = accident_cleaned['Local_Authority_District'].value_counts().reset_index()
I6_LAD.columns = ['Local_Authority_District', 'Number_of_Accidents']
I6_LAD = I6_LAD.nlargest(10, 'Number_of_Accidents')

plt.figure(figsize=(10, 6))
sns.barplot(x='Local_Authority_District', y='Number_of_Accidents', data=I6_LAD, palette='dark')
plt.title('Accidents across different Local Authority Districts')
plt.xlabel('Local Authority (District)')
plt.ylabel('Accidents')
plt.show()

