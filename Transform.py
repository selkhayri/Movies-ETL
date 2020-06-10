#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the dependencies

import pandas as pd
import json
import numpy as np


# In[2]:


# Set the data file path
file_dir = "../data/"


# In[3]:


with open(f"{file_dir}wikipedia.movies.json",mode="r") as file:
    wiki_movies_raw = json.load(file)


# In[4]:


len(wiki_movies_raw)


# In[5]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[6]:


wiki_movies_df.head()


# In[7]:


wiki_movies_columns = wiki_movies_df.columns.to_list()
len(wiki_movies_columns)


# In[8]:


wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                and 'imdb_link' in movie
                and 'No. of episodes' not in movie]


# In[9]:


len(wiki_movies)


# In[10]:


wiki_df = pd.DataFrame(wiki_movies)


# In[11]:


def clean_movie(movie):
    movie = dict(movie)  # create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    
    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
            
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')
    
    return movie


# In[12]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[13]:


len(clean_movies)


# In[14]:


def f(name):
    def g(leng):
        return leng**2
    
    return g(len(name))

f("sami")


# In[15]:


def f(a):
    
    a[2] *= 5
    
    a.append("hello")
    return a

a = [1,2,3,4]
f(a)


# In[16]:


square = lambda x:x*x
    
square(5)


# In[17]:


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())
            
            
            
    


# In[18]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[19]:


wiki_movies_df.isna().sum()


# In[20]:


wiki_movies_df.count()


# In[21]:


[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[22]:


[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[23]:


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[24]:


wiki_movies_df.dtypes


# In[25]:


box_office = wiki_movies_df['Box office'].dropna() 


# In[26]:


len(box_office)


# In[27]:


def is_not_a_string(x):
    return type(x) != str


# In[28]:


box_office[box_office.map(is_not_a_string)]


# In[29]:


box_office[box_office.map(lambda x: type(x) != str)]


# In[30]:


some_list = ['One','Two','Three']
'Mississippi'.join(some_list)


# In[31]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[32]:


box_office


# In[33]:


import re


# In[34]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
# form_one = r'\$\d+(\.\d{1,2})? [mb]illion'


# In[35]:


box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[36]:


# form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
# form_one = r'\$\s*\d+(\.\d*)?\s*[mb]illi?on'


# In[37]:


box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[38]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
ser1 = box_office[box_office.str.contains(form_two, flags=re.IGNORECASE)]
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[39]:


# form_two = r'\$\d{1,3}((,\d{3})+)?'
form_two = r'\$\d{1,3}((?:,\d{3})+)'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[40]:


box_office[box_office.str.contains(form_two, flags=re.IGNORECASE)]


# In[41]:


for item in ser1:
    print(item)


# In[42]:


# form_two = r'\$\d{1,3}((?:,\d{3})+)'
form_two = r'\$\d{1,3}((\,\d{3})+)?(\.\d{1,2})?((?!( [mb]illion).)+)$'
# box_office.str.contains(form_two, flags=re.IGNORECASE).sum()
ser2 = box_office[box_office.str.contains(form_two, flags=re.IGNORECASE)]
for item in ser2:
        print(item)


# In[43]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[44]:


def parse_dollars_2(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    # if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
    if re.match(r'\$[^\d]*\d{1,3}((\,\d{3})+)?(\.\d+)?[^\d]*milli?on$', s, flags=re.IGNORECASE):
        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    # elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
    # elif re.match(r'\$\d{1,3}((\,\d{3})+)?(\.\d+)?\s*billi?on$', s, flags=re.IGNORECASE):    
    elif re.match(r'\$[^\d]*\d{1,3}(\.\d+)?[^\d]*( billi?on)', s, flags=re.IGNORECASE): 
        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    # elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):
    elif re.match(r'\$\s*\d{1,3}((\,\d{3})+)(?!\s[mb]illi?on)$',s, flags=re.IGNORECASE):
    
        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[45]:


parse_dollars("$21.4 million")


# In[46]:


parse_dollars("$21. million")


# In[47]:


parse_dollars_2("$21.4 million")


# In[48]:


parse_dollars_2("$21. million")


# In[49]:


parse_dollars("$21.4 billion")


# In[50]:


parse_dollars("$21. billion")


# In[51]:


parse_dollars_2("$21.4 billion")


# In[52]:


parse_dollars_2("$21. billion")


# In[53]:


parse_dollars("$8,667,684")


# In[54]:


parse_dollars("$8,667,684.")


# In[55]:


parse_dollars_2("$8,667,684")


# In[56]:


parse_dollars("$8,667,684.")


# In[57]:


parse_dollars("$57,718,089")


# In[58]:


parse_dollars_2("$57,718,089")


# In[59]:


wiki_movies_df["Box office"] = wiki_movies_df["Box office"].map(lambda x: ' '.join(x) if type(x) == list else x)


# In[60]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[61]:


# wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars_2)


# In[62]:


wiki_movies_df['box_office']


# In[63]:


form_one = r'\$\d{1,3}((,\d{3})+)?(\.\d{1,2})?$'
form_two = r'\$[^\d]*\d{1,3}(\.\d+)?[^\d]*( [mb]illi?on)'

wiki_movies_df['box_office_2'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars_2)


# In[64]:


wiki_movies_df.box_office = wiki_movies_df.box_office.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[65]:


wiki_movies_df.columns


# In[66]:


len(wiki_movies_df.loc[(wiki_movies_df.box_office != wiki_movies_df.box_office_2) & ((np.isnan(wiki_movies_df.box_office) == True) & (np.isnan(wiki_movies_df.box_office_2) == True))])


# In[67]:


len(wiki_movies_df.loc[(wiki_movies_df.box_office != wiki_movies_df.box_office_2)])


# In[68]:


series = wiki_movies_df.loc[(wiki_movies_df.box_office != wiki_movies_df.box_office_2) & ((np.isnan(wiki_movies_df.box_office) != True) | (np.isnan(wiki_movies_df.box_office_2) != True))][["Box office","box_office","box_office_2"]]
print(len(series))
for item in series.items():
    print(item)


# In[69]:


parse_dollars_2("$2.790 billion")


# In[70]:


budget = wiki_movies_df['Budget'].dropna()


# In[ ]:





# In[71]:


budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[72]:


budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[73]:


form_one_2 = r'\$\d{1,3}((,\d{3})+)?(\.\d{1,2})?$'
form_two_2 = r'\$[^\d]*\d{1,3}(\.\d+)?[^\d]*( [mb]illi?on)'


# In[74]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)


# In[75]:


no_match = budget[~matches_form_one & ~matches_form_two]
for item in no_match:
    print(item)


# In[76]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars_2)


# In[77]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[78]:


release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[79]:


# release_date = release_date.str.lower()


# In[80]:


release_date


# In[81]:


#re_full_month_name = r'(?:january|february|march|april|may|june|july|august|september|october|november|december)'
re_full_month_name = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'

# Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
# re_month_day_year = r'(' + re_full_month_name + r'\s\d{1,2},\s\d{4}).*'
re_month_day_year = re_full_month_name + r'\s\d{1,2},\s\d{4}'

# Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
#re_year_month_day = r'[^\(]+\(\s+(\d{4}.\d{2}.\d{2}).*'
re_year_month_day = r'\d{4}.\d{2}.\d{2}'

# Full month name, four-digit year (i.e., January 2000)
# re_month_year = r'(' + re_full_month_name + r'\s\d{4})'
re_month_year = re_full_month_name + r'\s\d{4}'

# Four-digit year
# re_four_digit_year = r'(\d{4})'
re_four_digit_year = r'\d{4}'

date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[82]:


release_date.str.extract(f'({re_month_day_year})')


# In[83]:


release_date.str.extract(f'({re_year_month_day})')


# In[84]:


release_date.str.extract(f'({re_month_year})')


# In[85]:


release_date.str.extract(f'({re_four_digit_year})')


# In[86]:


release_date.str.extract(f'({re_month_day_year}|{re_year_month_day}|{re_month_year}|{re_four_digit_year})', flags=re.IGNORECASE)


# In[87]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({re_month_day_year}|{re_year_month_day}|{re_month_year}|{re_four_digit_year})')[0], infer_datetime_format=True)
# wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[88]:


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[89]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[90]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[91]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[92]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE)!=True]


# In[93]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[94]:


running_time_extract


# In[95]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[96]:


running_time_extract


# In[97]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[98]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[99]:


# Load the Movielens data
kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv',low_memory=False)


# In[100]:


kaggle_metadata.dtypes


# In[101]:


kaggle_metadata[["budget","release_date","popularity","adult","id","video"]].head()


# In[102]:


kaggle_metadata['adult'].value_counts()


# In[103]:


kaggle_metadata.shape[0]


# In[104]:


kaggle_metadata = kaggle_metadata[kaggle_metadata.adult.isin(['True','False'])]


# In[105]:


kaggle_metadata.shape[0]


# In[106]:


kaggle_metadata = kaggle_metadata.drop("adult",axis=1)


# In[107]:


kaggle_metadata.columns


# In[108]:


kaggle_metadata['video'].value_counts()


# In[109]:


kaggle_metadata['video'].value_counts()


# In[110]:


kaggle_metadata.dtypes


# In[111]:


kaggle_metadata[kaggle_metadata['video']==True]


# In[ ]:





# In[112]:


len(kaggle_metadata[kaggle_metadata['video']==True][["video"]])


# In[113]:


# This step is wrong!!
# kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[114]:


kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[115]:


kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[116]:


# Load the ratings data
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# In[117]:


ratings.info(null_counts=True)


# In[118]:


ratings['timestamp']


# In[119]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[120]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[121]:


# Plot a histogram of the rating column
ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[122]:


movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[123]:


movies_df.columns


# In[124]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle
# running_time             runtime
# budget_wiki              budget_kaggle
# box_office               revenue
# release_date_wiki        release_date_kaggle
# Language                 original_language
# Production company(s)    production_companies  


# In[125]:


movies_df[['title_wiki','title_kaggle']]


# In[126]:


movies_df[~(movies_df.title_wiki == movies_df.title_kaggle)][["title_wiki","title_kaggle"]]


# In[127]:


movies_df[(movies_df.title_kaggle == '') | (movies_df.title_kaggle.isnull())]


# In[128]:


movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[129]:


movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[130]:


movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[131]:


movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[132]:


movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[133]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[134]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')][["release_date_wiki","release_date_kaggle"]]


# In[135]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index


# In[136]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[137]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[138]:


movies_df[movies_df['release_date_kaggle'].isnull()]


# In[139]:


movies_df['Language'].value_counts()


# In[140]:


movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[141]:


# For the Kaggle data, there are no lists, so we can just run value_counts() on it

movies_df['original_language'].value_counts(dropna=False)


# In[142]:


movies_df[['Production company(s)','production_companies']]


# In[143]:


movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[144]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[145]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# for i in range(movies_df.shape[1]):
#     print(movies_df.columns[i])

# In[146]:


for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    # if num_values == 1:
    print(f"{col} : {num_values}")


# In[147]:


movies_df['video'].value_counts(dropna=False)


# In[148]:


movies_df[movies_df['video'] == True]


# In[149]:


def forloop(col):
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 2:
        return col
    #print(f"{col} : {num_values}")
    


# In[150]:


# How could you replace the previous for loop with a list comprehension?

list(filter(None, [forloop(col) for col in movies_df.columns]))


# In[151]:


# reorder the columns to make the dataset easier to read 

movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[152]:


# rename the columns to be consistent

movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[153]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()
rating_counts


# In[154]:


rating_counts = rating_counts.rename({'userId':'count'},axis=1)
rating_counts


# In[155]:


rating_counts = rating_counts.pivot(index='movieId',columns='rating', values='count')
rating_counts


# In[156]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[157]:


rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[158]:


rating_counts


# In[159]:


movies_df.columns


# In[160]:


movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[161]:


movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[162]:


from sqlalchemy import create_engine


# In[163]:


from config import db_password


# db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"

# engine = create_engine(db_string)

# movies_df.to_sql(name='movies', con=engine)

# import time
# 
# rows_imported = 0
# # get the start_time from time.time()
# start_time = time.time()
# for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
#     print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
#     data.to_sql(name='ratings', con=engine, if_exists='append')
#     rows_imported += len(data)
# 
#     # add elapsed time to final print out
#     print(f'Done. {time.time() - start_time} total seconds elapsed')

# In[170]:


dct = [{"a":True,"b":"age"}]
type(dct)
df = pd.DataFrame(dct)
df.dtypes

