#!/usr/bin/env python
# coding: utf-8

# Import the dependencies
import json
import pandas as pd
import numpy as np
import re

# Load data directory from local config 
from local_config import data_dir

# Load dollar amount regexps from local config
from local_config import re_dollar_amount_1
from local_config import re_dollar_amount_2
from local_config import re_dollar_amount_3

# Load the date regexps from local config
from local_config import re_date_form_1
from local_config import re_date_form_2
from local_config import re_date_form_3
from local_config import re_date_form_4

def clean_movie(movie):
    
    # Phase I
    movie = dict(movie)  # create a non-destructive copy
    alt_titles = {}
    
    # scan the movie dict parameter for alternate title keys
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
          
        if key in movie:
            # add the alternate title keys into a new dict object, alt_titles
            alt_titles[key] = movie[key]
            
            # remove the key,value item from the movie dict
            movie.pop(key)
    
    # if any alternate titles dict has any items, add the alt_titles dict to the movie dict using key, 'alt_titles'
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    
    # Phase II - rename the keys in the movie dict to match those in the kaggle data
    
    # This inner function changes a column name from old_name and new_name
    #
    # Parameters:
    # old_name - old key name
    # new_name - new key name
    
    def change_column_name(old_name, new_name):
        # If the key old_name exists in the movie dict
        if old_name in movie:
            
            # place the value of movie[old_name] into movie[new_name] and delete the old_name item
            movie[new_name] = movie.pop(old_name)
    
    # Perform the following key name changes 
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

#-------------------------------------------------

def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    #if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):
    if re.match(re_dollar_amount_1, s, flags=re.IGNORECASE):
        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    # elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):
    elif re.match(re_dollar_amount_2, s, flags=re.IGNORECASE):
        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(re_dollar_amount_3, s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan

#-------------------------------------------------

def Pipeline(wiki_movies_raw,kaggle_data,ratings_data):
    # Convert the wiki_movies json data to a dataframe
    #wiki_movies_df = pd.DataFrame(wiki_movies_raw)

    # Get list of movie json objects which satisfy the following conditions:
    # - one of the keys, Director or 'Directed by,' is present
    # - the imdb_link key is present
    # - the 'No. of episodes' key is not present
    wiki_movies = [movie for movie in wiki_movies_raw
                if ('Director' in movie or 'Directed by' in movie)
                and 'imdb_link' in movie
                and 'No. of episodes' not in movie]

    # Load the wiki_movies json objects list into a dataframe
    #wiki_df = pd.DataFrame(wiki_movies)
    
    # Clean the movies in in the wiki_movies list

    clean_movies = [clean_movie(movie) for movie in wiki_movies]
    
    # Redefine wiki_movies_df using the clean_movies dict list
    wiki_movies_df = pd.DataFrame(clean_movies)
    
    # Extract the imdb_id from using regular expression
    wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.\
        extract(r'(tt\d{7})')
            
    # Remove duplicate entries in wiki_movies_df
    wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)

    # Keep only those columns which have less than 90% null values
    wiki_columns_to_keep = [column for column in wiki_movies_df.columns \
                            if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
    wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
    
    print(wiki_movies_df.shape)
    
    # Handle non-string (list) entries in wiki_movies_df['Box office']
    # --------------------------
    box_office = wiki_movies_df['Box office'].dropna()
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    
    wiki_movies_df['box_office'] = box_office.str.extract(f'({re_dollar_amount_1}|'\
                                                          f'{re_dollar_amount_2}|'\
                                                          f'{re_dollar_amount_3})', \
                                                          flags=re.IGNORECASE)[0].apply(parse_dollars)
       
    #for item in wiki_movies_df['box_office']:
    #    print(f"box_office ... {item}")
    
    # Handle non-string (list) entries in wiki_movies_df['Budget']
    # --------------------------    
    budget = wiki_movies_df['Budget'].dropna()
    budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
    
    wiki_movies_df['budget'] = budget.str.extract(f'({re_dollar_amount_1}|'\
                                                  f'{re_dollar_amount_2}|'\
                                                  f'{re_dollar_amount_3})', \
                                                 flags=re.IGNORECASE)[0].apply(parse_dollars)    
    
    ### Remove the Budget column from the wiki_movie_df dataframe 
    wiki_movies_df.drop('Budget', axis=1, inplace=True)
    
    # Handling the Release date column
    #---------------------------------
    release_date = wiki_movies_df['Release date'].\
        dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
        
    wiki_movies_df['release_date'] = \
        pd.to_datetime(release_date.str.extract(f'({re_date_form_1}|'\
                                                f'{re_date_form_2}|'\
                                                f'{re_date_form_3}|'\
                                                f'{re_date_form_4})')[0],\
                                               infer_datetime_format=True)
     
    # print(wiki_movies_df['release_date'])
    
    # Handle the Running time column
    # ------------------------------
    running_time = wiki_movies_df['Running time'].\
        dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
    
    running_time_extract = running_time.str.\
        extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
    
    running_time_extract = running_time_extract.\
        apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
    
    wiki_movies_df['running_time'] = running_time_extract.\
        apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
    
    # Drop the Running time column
    wiki_movies_df.drop('Running time', axis=1, inplace=True)
    
    print(wiki_movies_df['running_time'])
    
    return None


if __name__ == "__main__":
    # Extract the Movielens and ratings data
    movies_metadata = pd.read_csv(f'{data_dir}movies_metadata.csv',low_memory=False)
    ratings = pd.read_csv(f'{data_dir}ratings.csv')
    
    # Extract the Wikipedia data
    with open(f'{data_dir}/wikipedia.movies.json', mode='r') as file:
        wiki_movies_raw = json.load(file)
    
    Pipeline(wiki_movies_raw,movies_metadata,ratings)

