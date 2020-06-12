#!/usr/bin/env python
# coding: utf-8

# Import the dependencies
import json
import pandas as pd
import numpy as np
import re
import psycopg2 as psql
import time
from sqlalchemy import create_engine

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

# Load the database particulars
from config import db_user
from config import db_password
from local_config import db_server

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
    if re.match(re_dollar_amount_1, s, flags=re.IGNORECASE):
        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
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



def get_table_columns(table_name):
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # find columns in existing table
    sql = f"SELECT * FROM {table_name};"
    
    cur.execute(sql)
    
    movie_cols = [desc[0] for desc in cur.description]
    
    cur.close()
    conn.close()
    
    return movie_cols

def delete_from_table(table_name):
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # find columns in existing table
    
    sql = f"DELETE FROM {table_name};"
    cur.execute(sql)
    
    cur.close()
    conn.close()

def remove_cols(table_name, cols):
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # find columns in existing table
    
    for col in cols:
        sql = f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {col};"
        cur.execute(sql)
    
    cur.close()
    conn.close()

def add_cols(table_name, cols):
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # find columns in existing table
    
    for col in cols:
        sql = f"ALTER TABLE {table_name} ADD COLUMN {col} varchar(30);"
        cur.execute(sql)
    
    cur.close()
    conn.close()

def load_ratings_data():   
    engine = create_engine(f"postgres://{db_user}:{db_password}@" + db_server)
    
    rows_imported = 0
    # get the start_time from time.time()
    
    start_time = time.time()
    for data in pd.read_csv(f'{data_dir}ratings.csv', chunksize=1000000):
        print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)
    
        # add elapsed time to final print out
        print(f'Done. {time.time() - start_time} total seconds elapsed')

def insert_data(table_name,df):
    engine = create_engine(f"postgres://{db_user}:{db_password}@" + db_server)

    df.to_sql(name=table_name, con=engine, if_exists="append")
    
def load_movies_data(df):
    
    delete_from_table("movies")
    
    movie_cols = get_table_columns("movies")
    df_cols = df.columns
    
    to_remove = []
    for col in movie_cols:
        if col != "index" and not col in df_cols:
            to_remove.append(col)
            
    to_add = []
    for col in df_cols:
        if col not in movie_cols:
            to_add.append(col)
    
    
    if len(to_remove) > 0:    
        remove_cols("movies", to_remove)
    
    if len(to_add) > 0:
        add_cols("movies", to_add)
        
    insert_data("movies",df)
    
    return None


#-------------------------------------------------

def Pipeline(wiki_movies_raw,kaggle_metadata,ratings_data):
    # Convert the wiki_movies json data to a dataframe
    #wiki_movies_df = pd.DataFrame(wiki_movies_raw)

    ############
    #
    # Wikipedia data
    #
    ############

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
    
    ############
    #
    # Movies Kaggle data
    #
    ############
    
    # Handle the adult column
    # -----------------------
    
    # Filter out the rows where the value of the adult column is something
    # other than 'True' or 'False.'
    kaggle_metadata = kaggle_metadata[kaggle_metadata.adult.isin(['True','False'])]
    
    # Remove adult movies from the data set
    kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False']
    
    # Drop the adult column
    kaggle_metadata = kaggle_metadata.drop("adult",axis=1)

    
    # Handle the video column
    # -----------------------
    
    # Check the values in the video column
    distinct_values = kaggle_metadata["video"].unique()
    
    # Filter out the rows where the value of the video column is something
    # other than the boolean values of True and False.
    if len(distinct_values) > 2:
       kaggle_metadata = kaggle_metadata[kaggle_metadata.video.isin([True,False])]     
        
    # Convert the video column to boolean
    kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
    
 
    # Handle the budget column
    # -----------------------
    
    # Convert the budget column to integer
    kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
    
    # Handle the id and popularity columns
    # -------------------------------------
    
    # Convert the id and popularity columns to integer
    kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
    kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
    
    # Handle the release_date column
    # -----------------------
    
    # Convert the  release_date column to datetime 
    kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])
    
    ##########
    #
    # Ratings data
    #
    ##########
    
    # Convert the timestamp column to datetime
    ratings_data['timestamp'] = pd.to_datetime(ratings_data['timestamp'], unit='s')
    
    ##########
    #
    # Further processing
    #
    ###########
    
    nulls_wiki = wiki_movies_df.loc[(wiki_movies_df["title"] == '') | (wiki_movies_df["title"].isnull())]

    # Join the kaggle_metadata and wiki_movies_df dataframes on imdb_id. To
    # distinguish between similarly named columns from the two tables, attach
    # the suffice "_wiki" to the columns from the wiki_movies_df dataframe 
    # and "_kaggle" to the columns from the kaggle_metadata dataframe
    movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])    
      
    # Handle the title_kaggle and title_wiki columns. 
    # ---------------------------------------------------
    
    # Determine the null values and empty title values that exist in the
    # title_kaggle column and the title_wiki column
    nulls_kaggle = movies_df.loc[(movies_df["title_kaggle"] == '') | (movies_df["title_kaggle"].isnull())]
    nulls_wiki = movies_df.loc[(movies_df["title_wiki"] == '') | (movies_df["title_wiki"].isnull())]
    
    # Cross pollinate the values in the two columns, filling the empty or null
    # values in one column with the values in the other
    movies_df.loc[nulls_wiki.index,"title_wiki"] = movies_df.loc[nulls_wiki.index,"title_kaggle"]
    movies_df.loc[nulls_kaggle.index,"title_kaggle"] = movies_df.loc[nulls_kaggle.index,"title_wiki"]
        
    # drop the title_wiki column
    movies_df.drop(columns=["title_wiki"])

    # Delete any rows with no title
    null_titles = movies_df.loc[(movies_df["title_kaggle"] == '') | (movies_df["title_kaggle"].isnull())]
    movies_df.drop(null_titles.index,axis=0)

    # Handle the running_time and runtime columns
    # -------------------------------------------
    
    # Fill the missing values in the kaggle column, runtime, with zeroes
    movies_df.runtime.fillna(0, inplace=True)
    
    # Get the rows where budget_kaggle is 0
    zeroes_runtime_kaggle = movies_df.loc[movies_df.runtime == 0]
    
     # Fill in the zeroes with the values from Wikipedia data
    movies_df.loc[zeroes_runtime_kaggle.index,"runtime"] = movies_df.loc[zeroes_runtime_kaggle.index,"running_time"]
    
    # Drop the wikipedia column, running_time
    movies_df.drop("running_time",axis=1,inplace=True)
    
    # Handle the budget_kaggle and budget_wiki columns
    # ------------------------------------------------
    
    # Fill the missing values in the kaggle column, runtime, with zeroes
    movies_df.budget_kaggle.fillna(0, inplace=True)
    
    # Get the rows where budget_kaggle is 0
    zeroes_budget_kaggle = movies_df.loc[movies_df.budget_kaggle == 0]
    
    # Fill in the zeroes with the values from Wikipedia data
    movies_df.loc[zeroes_budget_kaggle.index,"budget_kaggle"] = movies_df.loc[zeroes_budget_kaggle.index,"budget_wiki"]
    
    # Drop the wikipedia column, budget_wiki
    movies_df.drop("budget_wiki",axis=1,inplace=True)
    
    # Handle the box_office and revenue columns
    # -----------------------------------------
    
    # Fill the missing values in the kaggle column, box_office, with zeroes
    movies_df.revenue.fillna(0, inplace=True)
    
    # Get the rows where box_office is 0
    zeroes_revenue = movies_df.loc[movies_df.revenue == 0]
    
    # Fill in the zeroes with the values from Wikipedia data
    movies_df.loc[zeroes_revenue.index,"revenue"] = movies_df.loc[zeroes_revenue.index,"box_office"]
    
    # Drop the wikipedia column, budget_wiki
    movies_df.drop("box_office",axis=1,inplace=True)
    
    
    # Handle release_date_kaggle and release_date_wiki
    # ------------------------------------------------
    
    
    
    # Handle Language - Drop Wikipedia
    # --------------------------------
    movies_df.drop("Language",axis=1,inplace=True)
    
    # Handle Production Companies - Drop Wikipedia
    # --------------------------------------------
    movies_df.drop("Production company(s)",axis=1,inplace=True)
    
    # Reorder the columns (and remove the 'video' column which contains only 
    # one value)
    # --------------------
    new_order = ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]
    
    movies_df = movies_df[new_order]
     
     
   # Rename the columns
    # ------------------
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
    
    
    # Transform and merge rating data
    # -------------------------------
    
    # Group ratings by movieId and rating
    rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()
    
    # Rename the 'userId' column to 'count'
    rating_counts = rating_counts.rename({'userId':'count'},axis=1)
    
    # Create a pivot-table with movieId down the left, rating on top, and count
    # as the values in the table
    rating_counts = rating_counts.pivot(index='movieId',columns='rating', values='count')
    
    # Prepend the prefix, 'rating_', to the column names
    rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]
    
    # Merge the movies_df dataframe with the rating_counts dataframe
    movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
    
    # Fill missing ratings with 0
    movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)
    
    # Save data into DB
    # -----------------
    
    load_movies_data(movies_df)
    load_ratings_data()
    
    return None


if __name__ == "__main__":
    # Extract the Movielens and ratings data
    movies_metadata = pd.read_csv(f'{data_dir}movies_metadata.csv',low_memory=False)
    ratings = pd.read_csv(f'{data_dir}ratings.csv')
    
    # Extract the Wikipedia data
    with open(f'{data_dir}/wikipedia.movies.json', mode='r') as file:
        wiki_movies_raw = json.load(file)
    
    Pipeline(wiki_movies_raw,movies_metadata,ratings)

