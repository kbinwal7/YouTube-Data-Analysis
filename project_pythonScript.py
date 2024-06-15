# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ```Key Features:```
# 
# •	rank: Position of the YouTube channel based on the number of subscribers
# 
# •	Youtuber: Name of the YouTube channel
# 
# •	subscribers: Number of subscribers to the channel
# 
# •	video views: Total views across all videos on the channel
# 
# •	category: Category or niche of the channel
# 
# •	Title: Title of the YouTube channel
# 
# •	uploads: Total number of videos uploaded on the channel
# 
# •	Country: Country where the YouTube channel originates
# 
# •	Abbreviation: Abbreviation of the country
# 
# •	channel_type: Type of the YouTube channel 
# 
# •	video_views_rank: Ranking of the channel based on total video views
# 
# •	country_rank: Ranking of the channel based on the number of subscribers within its country
# 
# •	channel_type_rank: Ranking of the channel based on its type video_views_for_the_last_30_days: Total video views in the last 30 days
# 
# •	lowest_monthly_earnings: Lowest estimated monthly earnings from the channel
# 
# •	highest_monthly_earnings: Highest estimated monthly earnings from the channel
# 
# •	lowest_yearly_earnings: Lowest estimated yearly earnings from the channel
# 
# •	highest_yearly_earnings: Highest estimated yearly earnings from the channel
# 
# •	subscribers_for_last_30_days: Number of new subscribers gained in the last 30 days
# 
# •	created_year: Year when the YouTube channel was created
# 
# •	created_month: Month when the YouTube channel was created
# 
# •	created_date: Exact date of the YouTube channel's creation
# 
# •	Gross tertiary education enrollment (%): Percentage of the population enrolled in tertiary education in the country
# 
# •	Population: Total population of the country
# 
# •	Unemployment rate: Unemployment rate in the country
# 
# •	Urban_population: Percentage of the population living in urban areas
# 
# •	Latitude: Latitude coordinate of the country's location
# 
# •	Longitude: Longitude coordinate of the country's location
# 
# 

# %%
url = './data_Youtube.csv'
data_frame = pd.read_csv(url, index_col='rank', encoding='latin-1')
data_frame.head()
print(data_frame.isnull().sum())
colr=sns.color_palette('husl')
sns.set_palette('husl')

# %%
#removing rows with any missing value can reduce the data drastically like from 1000 instances to around 500 instances
#hence we have to handle missing values by assigning them some suitable values
#for numerical values we can use median(not mean because outliers exist)
#for categorical data we can use mode

data_frame['category'].fillna(data_frame['category'].mode()[0],inplace=True)
data_frame['Country of origin'].fillna(data_frame['Country of origin'].mode()[0],inplace=True)
data_frame['Country'].fillna(data_frame['Country'].mode()[0],inplace=True)
data_frame['channel_type'].fillna(data_frame['channel_type'].mode()[0],inplace=True)


data_frame['created_year'].fillna(data_frame['created_year'].mode()[0],inplace=True)
data_frame['created_month'].fillna(data_frame['created_month'].mode()[0],inplace=True)
data_frame['created_date'].fillna(data_frame['created_date'].mode()[0],inplace=True)

data_frame['Latitude'].fillna(data_frame['Latitude'].mode()[0],inplace=True)
data_frame['Longitude'].fillna(data_frame['Longitude'].mode()[0],inplace=True)
data_frame['video_views_for_the_last_30_days'].fillna(data_frame['video_views_for_the_last_30_days'].median(),inplace=True)
data_frame['subscribers_for_last_30_days'].fillna(data_frame['subscribers_for_last_30_days'].median(),inplace=True)
data_frame['Population'].fillna(data_frame['Population'].median(),inplace=True)
data_frame['subscribers'].fillna(data_frame['subscribers'].median(),inplace=True)

data_frame['Unemployment rate'].fillna(data_frame['Unemployment rate'].median(),inplace=True)
data_frame['Urban_population'].fillna(data_frame['Urban_population'].median(),inplace=True)
data_frame['Unemployment rate'].fillna(data_frame['Unemployment rate'].median(),inplace=True)
data_frame['Gross tertiary education enrollment (%)'].fillna(data_frame['Gross tertiary education enrollment (%)'].median(),inplace=True)

# %%
# Generate the histogram plots for the DataFrame with specified figure size
axList = data_frame.hist(bins=25, figsize=(10, 12))

# Loop through the axes to set the labels
for ax in axList.flatten():
    # Set x-label for the last row
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel('')
    
    # Set y-label for the first column
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel('count')

# Adjust layout to fit labels and subplots nicely
plt.tight_layout()
plt.show()


# %%
print(data_frame.isnull().sum())

# %% [markdown]
# 1.	What are the top 10 YouTube channels based on the number of subscribers?

# %%
#top 10 youtube channels
top_10=data_frame[:10]
print(top_10['Youtuber'])
print(top_10['category'].value_counts())
print(top_10['Country'].value_counts())

# %% [markdown]
# 2.	Which category has the highest average number of subscribers?

# %%
#category with highest number of subscribers
mid_s=data_frame['subscribers'].median()
print(data_frame['subscribers'].describe())
sum_by_category = data_frame.groupby('category')['subscribers'].sum().sort_values()
sum_by_type = data_frame.groupby('channel_type')['subscribers'].sum().sort_values()
#print(sum_by_category)
fig,ax=plt.subplots(1,2,figsize=(10,5))
sum_by_category.plot(kind='bar',color=colr,ax=ax[0])
ax[0].axhline(mid_s,color='red')
sum_by_type.plot(kind='bar',color=colr,ax=ax[1])


plt.suptitle("Number of Subscribers vs Category,type")
ax[0].set_ylabel('No of Subscribers')
ax[1].axhline(mid_s,color='red')
plt.tight_layout()
plt.grid()
plt.show()
sorted=sum_by_category.sort_values(ascending=False)
#print(sorted)
print("Maximum Subscribers for:", sorted.index[0],"; with :", sorted.iloc[0],"Subscribers")

# %% [markdown]
# 3.	How many videos, on average, are uploaded by YouTube channels in each category?

# %%
#category with highest number of average uploads
avg_category=data_frame.groupby('category')['uploads'].mean()

avg_category.plot(kind='bar',color=colr)
plt.grid()
sorted=avg_category.sort_values(ascending=False)
print(sorted)
print("Maximum uploads for:", sorted.index[0],";\nwith avg uploads of:", sorted.iloc[0])
#maximum view by News ans Politics


# %%
data_frame['category'].value_counts()
plt.figure(figsize=(10, 8))
sns.boxplot(x='category', y='uploads', data=data_frame)
plt.xlabel('Channel Category')
plt.ylabel('Uploads')
plt.title('Distribution of Uploads Across Different Channel Types')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
plt.grid(True)  # Add grid for better visualization of quartiles

plt.show()

# %%
sns.boxplot(x='uploads', data=data_frame, orient='h')
plt.title('Box Plot of Number of Uploads in Last 30 Days')
plt.xlabel('Uploads')
plt.yticks([])

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# 4.	What are the top 5 countries with the highest number of YouTube channels?
# 

# %%
#category with highest number of youtube channels
country_youtubers = data_frame['Country of origin'].value_counts().sort_values().tail(10)

# Plot the bar chart
country_youtubers.plot(kind='bar', color=colr)  # Assuming 'blue' as the color, you can change as needed
plt.xlabel('Country')
plt.ylabel('Number of YouTubers')
plt.title('Number of YouTubers by Country')

# Print the country with the maximum number of YouTubers
print("Maximum YouTubers from:", country_youtubers.index[9] ,"\nwith:", country_youtubers.iloc[0], "YouTubers")
plt.grid()
# Show the plot
plt.show()

# %%
us_data = data_frame[data_frame['Country of origin'] == 'United States']
#category_counts = us_data['category'].value_counts()
b_data = data_frame[data_frame['Country of origin'] == 'Brazil']
category_counts = b_data['category'].value_counts()
print("Brazil:",category_counts)


# %% [markdown]
# 5.	What is the distribution of channel types across different categories?
# 

# %%
#Distribution of channels across categories
channels_categories=data_frame['category'].value_counts()
channels_type=data_frame['channel_type'].value_counts()
fig,ax=plt.subplots(1,2,figsize=(10,5))
channels_categories.plot(kind='bar',color=colr,ax=ax[0])
channels_type.plot(kind='bar',color=colr,ax=ax[1])
plt.suptitle('Distribution of Youtube Channels across various Categories and Channel Types')

ax[0].set_ylabel("Number of Youtube Channels")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# 6.	Is there a correlation between the number of subscribers and total video views for YouTube channels?

# %%
x = data_frame['subscribers']
y = data_frame['video views']

new_data_frame = pd.DataFrame({'subscribers': x, 'video views': y})
new_data_frame = new_data_frame.drop_duplicates()

# Calculate correlation
corr_ = new_data_frame.corr()

# Plot scatterplot
sns.regplot(data=new_data_frame, x='subscribers', y='video views')
plt.xlabel('Subscribers')
plt.ylabel('Video Views')
plt.title('Scatterplot of Subscribers vs. Video Views')
plt.show()

print(corr_)
print("Yes, there is a positive correlation.")


# %% [markdown]
# 7.	How do the monthly earnings vary throughout different categories?
# 

# %%
#for lower bounfd of income grouped by category
lowest_cat = data_frame.groupby('category')['lowest_monthly_earnings'].mean()
#for upper bound of income grouped by category
highest_cat = data_frame.groupby('category')['highest_monthly_earnings'].mean()
#for average income grouped by category
avg = lowest_cat.add(highest_cat)
print("Monthly Mean Income by",avg.sort_values())

plt.plot(lowest_cat.index, lowest_cat.values, marker='o', label='Lowest Monthly Earnings')


plt.plot(highest_cat.index, highest_cat.values, marker='*', label='Highest Monthly Earnings')

plt.plot(avg.index, avg.values, marker='v', label='Average Monthly Earnings')


plt.xlabel('Category')
plt.ylabel('Mean Monthly Earnings')
plt.title('Mean Monthly Earnings by Category')


plt.legend()
plt.xticks(rotation=90)
plt.show()

# %%
avg.plot(kind='bar',color=colr)
plt.grid(axis='y')
plt.xlabel("Category")
plt.ylabel('Earnings')
plt.axhline(avg.median(),color='black',label='Median Income')
plt.legend()
plt.title('Earnings by category')
plt.show()



plt.figure(figsize=(8, 5))
sns.boxplot(x='category', y='highest_monthly_earnings', data=data_frame)
plt.xlabel('Category')
plt.ylabel('Highest Monthly Earnings')
plt.title('Distribution of Earning Across Categories')
plt.xticks(rotation=45) 
plt.grid(True) 

plt.show()

# %% [markdown]
# 8.	What is the overall trend in subscribers gained in the last 30 days across all channels?
# 

# %%
#mean_subscribers_by_category groups number of subscribers according to the categories and finds their mean
mean_subscribers_by_category = data_frame.groupby('category')['subscribers_for_last_30_days'].mean().reset_index()
print(mean_subscribers_by_category.sort_values(by='subscribers_for_last_30_days'))

mean_subscribers_by_category = mean_subscribers_by_category.sort_values(by='subscribers_for_last_30_days')


plt.figure(figsize=(12, 6))
plt.bar(mean_subscribers_by_category['category'], mean_subscribers_by_category['subscribers_for_last_30_days'], color='skyblue')
plt.plot(mean_subscribers_by_category['category'], mean_subscribers_by_category['subscribers_for_last_30_days'], marker='o', color='red')


plt.xlabel('Category')
plt.ylabel('Mean Subscribers for Last 30 Days')
plt.title('Mean Subscribers for Last 30 Days by Category')
plt.xticks(rotation=45) 
plt.grid(True)  # Add grid for better visualization of data points

# Show plot
plt.tight_layout()
plt.show()


# %% [markdown]
# 9.	Are there any outliers in terms of yearly earnings from YouTube channels?

# %%
#calculation of lower and upper fence values
#data_frame['lowest_yearly_earnings'].isnull().sum()
#data_frame['highest_yearly_earnings'].isnull().sum()
#no null alues are present
data_frame['average_yearly_earnings']=(data_frame['lowest_yearly_earnings']+data_frame['highest_yearly_earnings'])/2
data_frame['average_yearly_earnings'].describe()

minE=data_frame['average_yearly_earnings'].min()
Q1=data_frame['average_yearly_earnings'].quantile(0.25)
Q2=data_frame['average_yearly_earnings'].quantile(0.50)
Q3=data_frame['average_yearly_earnings'].quantile(0.75)
maxE=data_frame['average_yearly_earnings'].max()

#calculation for the average column
lower_fence=Q1-1.5*(Q3-Q1)
upper_fence=Q3+1.5*(Q3-Q1)

print("Upper Fence:",upper_fence,"Lower_Fence:",lower_fence)
#oultiers i=on left side not possible
#outliers on right side may exist

#print count of ouliers
count=(data_frame['average_yearly_earnings']>upper_fence).sum()
print("Number of ouliers on right Side",count)

# %%
fig, ax = plt.subplots(figsize=(8, 6))

# Create box plots for 'lowest_yearly_earnings', 'highest_yearly_earnings', and 'average_yearly_earnings'
box1 = ax.boxplot(data_frame['lowest_yearly_earnings'], positions=[1], widths=0.5, vert=False, patch_artist=True, boxprops=dict(facecolor='red'))
box2 = ax.boxplot(data_frame['highest_yearly_earnings'], positions=[2], widths=0.5, vert=False, patch_artist=True, boxprops=dict(facecolor='green'))
box3 = ax.boxplot(data_frame['average_yearly_earnings'], positions=[3], widths=0.5, vert=False, patch_artist=True, boxprops=dict(facecolor='brown'))

# Set labels and title and legend
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Lowest Yearly Earnings', 'Highest Yearly Earnings', 'Average Yearly Earnings'])
ax.set_xlabel('Earnings')
ax.set_title('Box Plots of Lowest, Highest, and Average Yearly Earnings')
ax.legend([box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]], ['Lowest', 'Highest', 'Average'], loc='upper right')

plt.show()


# %% [markdown]
# 10.	What is the distribution of channel creation dates? Is there any trend over time?

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.set_palette('husl')

# Plot histograms with KDE for each column, dropping NA values
sns.histplot(data_frame['created_date'].dropna(), kde=True, bins=10, ax=ax[0])
sns.histplot(data_frame['created_month'].dropna(), kde=True, bins=10, ax=ax[1])
sns.histplot(data_frame['created_year'].dropna(), kde=True, bins=10, ax=ax[2])


ax[0].set_xlabel('Created Date')
ax[0].set_ylabel('Frequency')
ax[1].set_xlabel('Created Month')
ax[2].set_xlabel('Created Year')


plt.tight_layout()
plt.show()

'''data_frame['created_date'].value_counts().sort_values().plot(kind='barh', ax=ax[0],color=colr)
ax[0].set_title('Counts by Date')

data_frame['created_month'].value_counts().sort_values().plot(kind='barh', ax=ax[1],color=colr)
ax[1].set_title('Counts by Month')

data_frame['created_year'].value_counts().sort_values().plot(kind='barh', ax=ax[2],color=colr)
ax[2].set_title('Counts by Year')'''




# %% [markdown]
# 11.	Is there a relationship between gross tertiary education enrollment and the number of YouTube channels in a country?
# 

# %%
print("Null values in GTEE(%):",data_frame['Gross tertiary education enrollment (%)'].isnull().sum())
#null values exist in the gross tertiary education enrollment column,
#we have to handle the missing values, we should now replace the missing values by 
# any central tendency depending upon the distribution of column

'''PLOTTING THE HISTOGRAM TO KNOW THE DISTRIBUTION'''
plt.figure(figsize=(10, 6))
sns.histplot(data_frame['Gross tertiary education enrollment (%)'].dropna(), kde=True, bins=10)
mean_value = data_frame['Gross tertiary education enrollment (%)'].mean()
median_value = data_frame['Gross tertiary education enrollment (%)'].median()
mode_value = data_frame['Gross tertiary education enrollment (%)'].mode()[0]

# Plot the mean, median, and mode on the histogram
plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
plt.axvline(median_value, color='green', linestyle='-', label=f'Median: {median_value:.2f}')
plt.axvline(mode_value, color='blue', linestyle='-.', label=f'Mode: {mode_value:.2f}')

# Add titles and labels
plt.title('Distribution of Gross tertiary education enrollment')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()



'''Lets replace null values with the median'''
data_frame['Gross tertiary education enrollment (%)'].fillna(median_value)
country_youtube=data_frame['Country of origin'].value_counts().sort_index()
#print(country_youtube)
print(data_frame['Gross tertiary education enrollment (%)'].isnull().sum())

# %%
#ccraete annew data frame for calculation of correlation
country_gtee = data_frame[['Country of origin', 'Gross tertiary education enrollment (%)']]
country_gtee = country_gtee.drop_duplicates().set_index('Country of origin').sort_index().drop_duplicates()


country_youtube = data_frame['Country of origin'].value_counts().sort_index().drop_duplicates()


no_youvsGtee = pd.concat([country_gtee, country_youtube], axis=1)

# Rename columns for clarity
no_youvsGtee.columns = ['Gross tertiary education enrollment (%)', 'YouTube Count']

# Plot the data
plt.figure(figsize=(10, 6))
sns.regplot(x='YouTube Count', y='Gross tertiary education enrollment (%)', data=no_youvsGtee)

# Add titles and labels
plt.title('Relationship between YouTube Count and Gross Tertiary Education Enrollment (%)')
plt.xlabel('YouTube Count')
plt.ylabel('Gross Tertiary Education Enrollment (%)')
plt.show()

# Calculate and print the correlation coefficient
corr_matrix = no_youvsGtee.corr()
corr_factor = corr_matrix.loc['YouTube Count', 'Gross tertiary education enrollment (%)']
print(f'Correlation coefficient: {corr_factor}')


# %% [markdown]
# 12.	How does the unemployment rate vary among the top 10 countries with the highest number of YouTube channels?

# %%
print("Null values in Unemployemnt rate column:" ,data_frame['Unemployment rate'].isnull().sum())
# Null values exist -> replace with median(already done above)
median_val = data_frame['Unemployment rate'].median()
data_frame['Unemployment rate'].fillna(median_val)

unemployment_ = data_frame.set_index('Country of origin')['Unemployment rate'].sort_index()

# Set index of youtube_count to match unemployment_
youtube_count = data_frame['Country of origin'].value_counts().sort_index()

# Reindex youtube_count to align with unemployment_
youtube_count = youtube_count.reindex(unemployment_.index)

unemploymentVSCountry = pd.concat([unemployment_, youtube_count], axis=1)
unemploymentVSCountry.columns = ['Unemployment rate', 'YouTube Count']  # Rename columns for clarity
unemploymentVSCountry=unemploymentVSCountry.drop_duplicates()
#print(unemploymentVSCountry)

sorted_unemploymentVSCountry = unemploymentVSCountry.sort_values(by='YouTube Count', ascending=False)
#print(sorted_unemploymentVSCountry)
first_10_unemployment_rates = sorted_unemploymentVSCountry['Unemployment rate'].head(10)

# Plot the first 10 unemployment rates
plt.figure(figsize=(10, 6))
plt.bar(first_10_unemployment_rates.index, first_10_unemployment_rates,color=colr)
plt.title('First 10 Unemployment Rates')
plt.xlabel('Country of Origin')
plt.ylabel('Unemployment Rate')
plt.xticks(rotation=45)
plt.show()

first_10_unemployment_rates.describe()


# %% [markdown]
# 13.	What is the average urban population percentage in countries with YouTube channels?
# 

# %%
data_frame.columns
#urban_population_avg=data_frame.groupby('Country of origin')['Urban_population'].mean().dropna()

data_frame['Urban_population(%)'] = (data_frame['Urban_population'] / data_frame['Population']) * 100

# Select only 'Urban_population(%)' and 'Country of origin', drop duplicates
result = data_frame[['Country of origin', 'Urban_population(%)']].drop_duplicates().dropna()

#print(result)
print(result.describe())

plt.figure(figsize=(8, 6))
plt.boxplot(result.iloc[:, 1], vert=False)
plt.title('Box Plot of Urban Population Percentage')
plt.xlabel('Urban Population Percentage')
plt.yticks([])
plt.show()

#ouliers do exist
#avergae urban population is 70%

# %%
data_frame.columns

# %% [markdown]
# 14.	Are there any patterns in the distribution of YouTube channels based on latitude and longitude coordinates?
# 

# %%
channels_df_unique = data_frame.drop_duplicates(subset=['Latitude', 'Longitude'])

'''# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(channels_df_unique['Longitude'], channels_df_unique['Latitude'], s=10, alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribution of YouTube Channels')
plt.grid(True)
plt.show()'''

# Heatmap plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=channels_df_unique, x='Longitude', y='Latitude', cmap='Blues', fill=True, thresh=0.05)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Heatmap: Distribution of YouTube Channels')
plt.show()


# %%
data_frame['Latitude'].describe()
data_frame['Longitude'].describe()

# %% [markdown]
# 15.	What is the correlation between the number of subscribers and the population of a country

# %%
sub_country = data_frame.groupby('Country of origin')['subscribers'].mean()
median_val = data_frame['Population'].median()
data_frame['Population'].fillna(median_val)
pop_country = data_frame.groupby('Country of origin')['Population'].mean().drop_duplicates().sort_index()
merged_data = pd.concat([sub_country, pop_country], axis=1)
merged_data.columns = ['Mean Subscribers', 'Mean Population']  # Rename columns for clarity

# Check for missing values
missing_values = merged_data.isnull().sum()
print("Missing values in merged_data:\n", missing_values)


merged_data.fillna(merged_data.mean())

# Calculate correlation matrix
corr_matrix = merged_data.corr()
print("Correlation Matrix:\n", corr_matrix)
sns.heatmap(corr_matrix)
            
plt.figure(figsize=(8, 6))
sns.regplot(x='Mean Population', y='Mean Subscribers', data=merged_data, scatter_kws={'s': 100})
plt.title('Regression Plot of Mean Subscribers vs Mean Population')
plt.xlabel('Mean Population')
plt.ylabel('Mean Subscribers')
plt.show()
#very weak negative correlation exists


# %% [markdown]
# 16.	How do the top 10 countries with the highest number of YouTube channels compare in terms of their total population?

# %%
pop_country=data_frame.groupby('Country')['Population'].mean().dropna().sort_index()
channel_count=data_frame['Country'].value_counts().sort_index()
merged_data=pd.concat([pop_country,channel_count],axis=1)
merged_data.columns=['Population','No of channels']
merged_data=merged_data.sort_values(by='No of channels')
#print(merged_data)


top10=merged_data.tail(10)
print(top10.describe())
plt.figure(figsize=(10, 6))
plt.bar(top10.index, top10['Population'], color=colr)
plt.axhline(merged_data['Population'].median(),color='black',label='Median Population')
plt.xlabel('Country')
plt.ylabel('Population')
plt.title('Population of Top 10 Countries with Highest Number of YouTube Channels')
plt.xticks(rotation=45)
plt.legend()
plt.grid()


# %% [markdown]
# ```**17.	Is there a correlation between the number of subscribers gained in the last 30 days and the unemployment rate in a country?**```

# %%
# Grouping by 'Country of origin' and calculating mean subscribers
sub_country = data_frame.groupby('Country of origin')['subscribers'].mean()

# Filling missing values in 'Unemployment rate' with median
median_val = data_frame['Unemployment rate'].median()
data_frame['Unemployment rate'].fillna(median_val, inplace=True)

# Grouping by 'Country of origin' and calculating mean unemployment rate
UR_country = data_frame.groupby('Country of origin')['Unemployment rate'].mean().sort_index()

# Concatenating the two series into a DataFrame
merged_data = pd.concat([sub_country, UR_country], axis=1)
merged_data.columns = ['Mean Subscribers', 'Mean UR']  # Renaming columns for clarity

# Check for missing values in merged_data
missing_values = merged_data.isnull().sum()
print("Missing values in merged_data:\n", missing_values)

# Handle missing values (if any) - this line does not actually fill missing values permanently
merged_data.fillna(merged_data.mean(), inplace=True)

# Calculate correlation matrix
corr_matrix = merged_data.corr()
print("Correlation Matrix:\n", corr_matrix)

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Plot regression plot (regplot)
plt.figure(figsize=(8, 6))
sns.regplot(x='Mean UR', y='Mean Subscribers', data=merged_data, scatter_kws={'s': 100})
plt.title('Regression Plot of Mean Subscribers vs Mean Unemployment Rate')
plt.xlabel('Mean Unemployment Rate')
plt.ylabel('Mean Subscribers')
plt.show()


# %% [markdown]
# ```**18.	How does the distribution of video views for the last 30 days vary across different channel types?**```

# %%
views_by_category = data_frame.groupby('channel_type')['video views'].sum()
views_by_category.plot(kind='bar', color=['orange', 'green'])

plt.title('Total Video Views by Type')
plt.xlabel('Channel Type')
plt.ylabel('Total Video Views')
plt.grid(axis='y')
plt.show()


views_by_category.describe()


# %%
plt.figure(figsize=(10, 8))
#make a box plot to visualise outliers in each categories better
sns.boxplot(x='channel_type', y='video_views_for_the_last_30_days', data=data_frame)
plt.xlabel('Channel Type')
plt.ylabel('Video Views for Last 30 Days')
plt.title('Distribution of Video Views Across Different Channel Types')
plt.xticks(rotation=45) 
plt.grid(True)  

plt.show()


# %% [markdown]
# ```**19.	Are there any seasonal trends in the number of videos uploaded by YouTube channels?**```

# %%
uploads_by_category = data_frame.groupby('category')['uploads'].mean()

uploads_year=data_frame.groupby('created_year')['uploads'].mean()
plt.plot(years, uploads_year, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)
plt.xlabel("Year")
plt.ylabel('Number of Uploads')
plt.axhline(uploads_year.median(),color='black',linestyle='--',label='Median Uploads')
plt.title("Number of uploads by Year")
uploads_by_category.describe()
print(uploads_year)

# %%
g = sns.FacetGrid(data_frame, col='category', col_wrap=3, height=4, aspect=1.5)
g.map(sns.lineplot, 'created_date', 'uploads')

g.set_axis_labels('Date', 'Uploads')
g.set_titles(col_template='{col_name}')
print("Trends by created Date")

# Rotate x-axis labels for better readability
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# %%
g = sns.FacetGrid(data_frame, col='category', col_wrap=3, height=4, aspect=1.5)

# Map the line plot to the grid
g.map(sns.lineplot, 'created_year', 'uploads')

# Add axis labels and titles
g.set_axis_labels('Year', 'Uploads')
g.set_titles(col_template='{col_name}')
print("Trend by created year")
# Rotate x-axis labels for better readability
for ax in g.axes.flatten():
    ax.tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()

# %% [markdown]
# ```***20.	What is the average number of subscribers gained per month since the creation of YouTube channels till now?**```

# %%
import numpy as np
avg_subscribers_per_year=data_frame.groupby('created_year')['subscribers_for_last_30_days'].mean()
# Extracting years and average values
years = avg_subscribers_per_year.index
avg_values = avg_subscribers_per_year.values


plt.figure(figsize=(10, 6))
plt.plot(years, avg_values, marker='o', color='skyblue', linestyle='-', linewidth=2, markersize=8)


median_subscribers = np.median(avg_values)
plt.axhline(median_subscribers, color='red', linestyle='--', label=f'Median Subscribers: {median_subscribers:.2f}')


plt.xlabel('Year')
plt.ylabel('Average Subscribers for Last 30 Days')
plt.title('Average Subscribers per Year')
plt.legend()  
plt.grid(True)
plt.show()



