import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from ydata_profiling import ProfileReport

import plotly.graph_objects as go

import statistics

# Read dataset and display the head
df = pd.read_csv('netflix_daily_top10.csv')
print("--- Here is the head of the dataset: \n", df.head())

# Drop duplicate rows
df = df.drop_duplicates()

# Display dataframe shape and information
print("\n --- The dataframe has a shape of: ", df.shape)
print("\n --- The different columns have these types: ", df.info())

# Check and replace null values
print("\n --- Check how many null values are present: ", df.isnull().sum())
df['Netflix Exclusive'].fillna('NO', inplace=True)
df['Year to Date Rank'] = df['Year to Date Rank'].replace('-', 0).astype('int64')
df['Last Week Rank'] = df['Last Week Rank'].replace('-', 0).astype('int64')

# Convert columns to appropriate date format
df['Netflix Release Date'] = pd.to_datetime(df['Netflix Release Date'])
df['As of'] = pd.to_datetime(df['As of'])
print("\n --- After converting to adequate format, the different columns have these types: ", df.info())

# Generate descriptive statistics for the sample
print("\n --- Here are the descriptive statistics for the sample: ", df.describe())

#Look at the data with ydata_profiling report
print("\n --- Produce a full report and save it: ")
profiling = ProfileReport(df, title="Profiling Report")
profiling.to_file("profiling.html")

#Days In Top 10 is highly overall correlated with Last Week Rank and 1 other fields	High correlation
#Last Week Rank is highly overall correlated with Days In Top 10 and 1 other fields	High correlation
#Rank is highly overall correlated with Year to Date Rank	High correlation
#Viewership Score is highly overall correlated with Days In Top 10 and 1 other fields
#Year to Date Rank is highly overall correlated with Rank

# Verifing with scatter and regression plot the high correlations
plt.figure()
sns.scatterplot(data=df, x=df['Days In Top 10'], y=df['Viewership Score']).set(title='Days In Top 10 vs Viewership Score')
corr_day_score=np.corrcoef(df['Viewership Score'],df['Days In Top 10'])
plt.savefig("Plots/DaysInTop10vsViewershipScore.pdf", format='pdf')

plt.figure()
sns.regplot(data=df, x=df['Days In Top 10'], y=df['Viewership Score']).set(title='Regression Days In Top 10 vs Viewership Score')
plt.savefig("Plots/RegressionDaysInTop10vsViewershipScore.pdf", format='pdf')

plt.figure()
sns.scatterplot(data=df, x=df['Days In Top 10'], y=df['Last Week Rank']).set(title='Days In Top 10 vs Last Week Rank')
corr_day_weekrank=np.corrcoef(df['Last Week Rank'],df['Days In Top 10'])
plt.savefig("Plots/DaysInTop10vsLastWeekRank.pdf", format='pdf')

plt.figure()
sns.scatterplot(data=df, x=df['Rank'], y=df['Year to Date Rank']).set(title='Rank vs Year to Date Rank')
corr_yearrank_rank=np.corrcoef(df['Rank'],df['Year to Date Rank'])
plt.savefig("Plots/RankvsYeartoDateRank.pdf", format='pdf')

print(corr_day_score[0][1], corr_day_weekrank[0][1], corr_yearrank_rank[0][1])


# Using countplot to see which genre has performed best.
plt.figure()
sns.set_style('whitegrid')
sns.countplot(data=df,x=df['Type'])
plt.savefig("Plots/HistoType.pdf", format='pdf')

# Analyzing and displaying top TV shows and movies
tv_shows = df[df['Type'] == 'TV Show']
tv_shows_top = tv_shows.groupby('Title')['Days In Top 10'].max().sort_values(ascending=False)[:11]
print("\n --- The TV shows that stayed longer in the Top 10 are: \n", tv_shows_top)

movies = df[df['Type'] == 'Movie']
movies_top = movies.groupby('Title')['Days In Top 10'].max().sort_values(ascending=False)[:11]
print("\n --- The movies that stayed longer in the Top 10 are: \n", movies_top)

# Visualizing Netflix Exclusive shows/movies percentage using pie charts
def plot_pie_chart(data, title):
    values = data['Days In Top 10']
    fig = go.Figure(data=[go.Pie(labels=data.index, values=values, textinfo='label+percent', insidetextorientation='radial')])
    fig.update_layout(title=title)
    return fig


tv_report = tv_shows.groupby('Netflix Exclusive').count()
plot_pie_chart(tv_report, 'Percentage of Netflix Exclusive Shows in Top TV Shows')
plt.savefig("Plots/PieNetflixExclusive_Shows.pdf", format='pdf')

movie_report = movies.groupby('Netflix Exclusive').count()
plot_pie_chart(movie_report, 'Percentage of Netflix Exclusive Movies in Top Movies')
plt.savefig("Plots/PieNetflixExclusive_Movies.pdf", format='pdf')

# Analyzing stand-up comedy shows
stand_up = df[df['Type'] == 'Stand-Up Comedy']
stand_up_top = pd.DataFrame(stand_up.groupby('Title')['Days In Top 10'].max().sort_values(ascending=False))
stand_up_title = stand_up.groupby('Title').max().sort_values(by='Days In Top 10', ascending=False)
stand_up_report = stand_up_title.groupby('Netflix Exclusive').size()
stand_up_report_percentage = (stand_up_report / stand_up_report.sum()) * 100
print("\n --- The stand-up comedy that stayed longer in the Top 10 are: \n", stand_up_top)
print("And this is the percentage of how many of those were Netflix exclusive: \n", stand_up_report_percentage)


