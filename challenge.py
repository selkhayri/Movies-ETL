#!/usr/bin/env python
# coding: utf-8

# Import the dependencies
import json
import pandas as pd
import numpy as np
import re
import psycopg2 as psql
from psycopg2 import OperationalError, errorcodes, errors
from sqlalchemy import create_engine
import time
import logging
import logging.handlers
import sys

# Load data directory from local config 
from local_config import data_dir

# Load the log director from local config
from local_config import log_dir

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

# Set up a specific logger with our desired output level
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.INFO)

# Add the log message handler to the logger
LOG_FILENAME =  log_dir + 'log.dat'
handler = logging.handlers.RotatingFileHandler(
              LOG_FILENAME, maxBytes=20, backupCount=5)
my_logger.addHandler(handler)

# print_psycopg2_exception
# Based on print_psycopg2_exception
# Retrieved from:
# https://kb.objectrocket.com/postgresql/python-error-handling-with-the-psycopg2-postgresql-adapter-645
# This function handles and parses psycopg2 exceptions
# 
def log_psycopg2_exception(err):
    # get details about the exception
    err_type, err_obj, traceback = sys.exc_info()

    # get the line number when exception occured
    line_num = traceback.tb_lineno
    my_logger.error("\npsycopg2 ERROR:", err, "on line number:", line_num)
    
    # Log the error traceback and error type
    my_logger.error("psycopg2 traceback:", traceback, "-- type:", err_type)

    # Log extensions.Diagnostics object attribute
    my_logger.error ("\nextensions.Diagnostics:", err.diag)

    # Log the pgcode and pgerror exceptions
    my_logger.error ("pgerror:", err.pgerror)
    my_logger.error ("pgcode:", err.pgcode, "\n")

# clean_movie
# ===========
# This function consolidates all the alternate titles keys into a single
# key whose value is a dictionary which contain the values of the other
# columns as key-value pairs. The alternate titles items are then removed fro
# the original dictionary. It also changes the names of some of the keys to 
# match those in the kaggle data.
#
# Parameters:
# - movie:  A dictionary of the current column from the movies table
#

def clean_movie(movie):
    
    # Check parameter types
    if type(movie) != dict:
        raise ValueError("Invalid movie object: Expected dict")
        
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
        
        # Check parameter types
        if type(old_name) != str:
            raise ValueError("Invalid old_name: Expected <str>")
            
        if type(new_name) != str:
            raise ValueError("Invalid new_name: Expected <str>")
        
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

# parse_dollars:
# This function converts string monetary values into floats
#
# Parameter:
# s - string monetary value in one of the following formats:
#       - $###.### million
#       - $###.### billion
#       - $###,###,###
# Return:
# The float value of the passed string, eg/ 729000000.00, or NaN if the string
# cannot be converted into a float

def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.### million
    if re.match(re_dollar_amount_1, s, flags=re.IGNORECASE):
        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.### billion
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

# get_table_columns
# =================
# This function returns a list of the names of the columns of the table
# whose name is passed in as a parameter.
#
# Parameters:
# - table_name - string - the name of the table for which a list column name is 
#                         required.
#
# Return:
# - the list of column names for table table_name

def get_table_columns(table_name):
    
    # Check parameter types
    if type(table_name) != str:
        raise ValueError("Invalid table_name: Expected <str>")
    
    # Get a database connection
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
   
    # find columns of table table_name
    sql = f"SELECT * FROM {table_name};" 
    cur.execute(sql)
    
    # List comprehension - use cursor description of the table to retrieve 
    # column names
    movie_cols = [desc[0] for desc in cur.description]
    
    # Close the cursor and the connection
    cur.close()
    conn.close()
    
    return movie_cols


# delete_from _table
# ==================
# This function deletes the entire contents of the table whose name is passed
# in as the parameter, table_name
#
# Paramter:
# - table_name - string - the name of the table whose contents are to be cleared

def delete_from_table(table_name):

    # Check parameter types
    if type(table_name) != str:
        raise ValueError("Invalid table_name: Expected <str>")
        
    # Get a database connection
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # Delete the contents of table table_name
    sql = f"DELETE FROM {table_name};"
    cur.execute(sql)
    
    # Close the cursor and the connection
    cur.close()
    conn.close()


# remove_cols
# ===========
# This function drops the columns that are specified in the cols parameter
# from table table_name
#
# Parameters:
# - table_name - string - the name of the table from which to drop columns
# - cols - list - the list of columns to be dropped

def remove_cols(table_name, cols):
    
    # Check parameter types
    if type(table_name) != str:
        raise ValueError("Invalid table_name: Expected <str>")
    
    if type(cols) != list:
        raise ValueError("Invalid cols: Expected <list>")
    
    # Get a database connection
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # Remove columns cols from table table_name
    for col in cols:
        if type(col) != str:
            raise ValueError("Invalid column name: Expected <str>")
            
        sql = f"ALTER TABLE {table_name} DROP COLUMN IF EXISTS {col};"
        cur.execute(sql)
    
    # Close the cursor and the connection
    cur.close()
    conn.close()

# add_cols
# ========
# This function adds the columns specified in parameter, cols, to table 
# table_name. The limitation of this function is that all added columns
# must be of type string, varchar(30)
#
# Parameters
# - table_name - string - the name of the table to which columns are added
# - cols - list - the names of the columns to be added to table_name

def add_cols(table_name, cols):

    # Check parameter types
    if type(table_name) != str:
        raise ValueError("Invalid table_name: Expected <str>")
    
    if type(cols) != list:
        raise ValueError("Invalid cols: Expected <list>")    

    # Get a database connection
    conn = psql.connect(f"postgres://{db_user}:{db_password}@" + db_server)
    cur = conn.cursor()
    
    # Add the columns to table table_name
    for col in cols:
        sql = f"ALTER TABLE {table_name} ADD COLUMN {col} varchar(30);"
        cur.execute(sql)
    
    # Close the cursor and the connection
    cur.close()
    conn.close()

# load_ratings_data
# =================
# This function loads the ratings data from the ratings.csv file into the
# the ratings table.

def load_ratings_data():
    
    my_logger.info("Loading ratings table ...")
    
    # Get a database connection
    engine = create_engine(f"postgres://{db_user}:{db_password}@" + db_server)
    
    # Initialize the number of imported rows
    rows_imported = 0
    
    # get the start_time from time.time()
    start_time = time.time()
    
    # Read the contents for the ratings.csv file
    for data in pd.read_csv(f'{data_dir}ratings.csv', chunksize=1000000):
        
        # log range of imported rows
        my_logger.info(f'importing rows {rows_imported} to {rows_imported + len(data)}...')
        
        # Insert the contents into the ratings table    
        data.to_sql(name='ratings', con=engine, if_exists='append')
        rows_imported += len(data)
    
        # add elapsed time to final print out
        my_logger.info(f'Done. {time.time() - start_time} total seconds elapsed')
        
        
# insert_data
# ===========
# This function uses the Pandas Dataframe method, to_sql, populate table
# table_name with the contents of dataframe, df.
#
# Parameters:
# - table_name - string - name of the table to be populated
# - df - DataFrame - dataframe whose contents will populate table table_name

def insert_data(table_name,df):
    
    # Check parameter types
    if type(table_name) != str:
        raise ValueError("Invalid table_name: Expected <str>")
    
    if type(df) != pd.DataFrame:
        raise ValueError("Invalid df: Expected <pandas.DataFrame>")
    
    # Get a database connection
    engine = create_engine(f"postgres://{db_user}:{db_password}@" + db_server)
    
    # Append the contents of dataframe df into table table_name
    df.to_sql(name=table_name, con=engine, if_exists="append")

# load_movies_data
# ================
# This function accomplishes the following items:
# - delete the contents of the passed in dataframe
# - compares the columns in the movies table with those in the movies dataframe
# - Adds and removes columns to the movies table as necessary
# - Calls the insert_data function to insert the contents of the movie dataframe
#   into the movies table
# 
# Parameters:
# - df - DataFrame - the dataframe whose content is to be inserted in the movies
#                    table
#

def load_movies_data(df):
   
    my_logger.info("Loading movies table ...") 
    
    # Check parameter types
    if type(df) != pd.DataFrame:
        raise ValueError("Invalid df: Expected <pandas.DataFrame>")
   
    # Delete the contents of the movies table
    delete_from_table("movies")
    
    # Get the names of the columns in the movies table
    movie_cols = get_table_columns("movies")
    
    # Get the names of the columns in the movies dataframe
    df_cols = df.columns
    
    # Determine which columns need to be removed from the movies table
    to_remove = []
    for col in movie_cols:
        if col != "index" and not col in df_cols:
            to_remove.append(col)
    
    # Determine which columns need to be added to the movies table
    to_add = []
    for col in df_cols:
        if col not in movie_cols:
            to_add.append(col)
    
    # Remove the columns that need to be removed
    if len(to_remove) > 0:    
        remove_cols("movies", to_remove)
    
    # Add the columns that need to be added
    if len(to_add) > 0:
        add_cols("movies", to_add)
        
    # Insert the new data into the movies table
    insert_data("movies",df)
    

# Pipeline
# ========
# This function takes three arguments, a wikipedia dataframe, a kaggle 
# dataframe, and a ratings dataframe, and uses them to update the movies
# and ratings tables in the movie_data database. 
#
# Parameters
# - wiki_movies_raw - list - Wikipedia movie objects
# - kaggle_metadata - dataframe - kaggle
# - ratings_data - dataframe - kaggle

def Pipeline(wiki_movies_raw,kaggle_metadata,ratings_data):

    # Check parameter types
    if type(wiki_movies_raw) != list:
        raise ValueError("Invalid wiki_movies_raw: Expected <pandas.DataFrame>")
    
    if type(kaggle_metadata) != pd.DataFrame:
        raise ValueError("Invalid kaggle_metadata: Expected <pandas.DataFrame>")
        
    if type(ratings_data) != pd.DataFrame:
        raise ValueError("Invalid ratings_data: Expected <pandas.DataFrame>")

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
    
   
    # Handle non-string (list) entries in wiki_movies_df['Box office']
    # --------------------------
    box_office = wiki_movies_df['Box office'].dropna()
    box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
        
    wiki_movies_df['box_office'] = box_office.str.extract(f'({re_dollar_amount_1}|'\
                                                          f'{re_dollar_amount_2}|'\
                                                          f'{re_dollar_amount_3})', \
                                                          flags=re.IGNORECASE)[0].apply(parse_dollars)
       
    
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
    
    # Drop the wikipedia column, release_date_wiki
    movies_df.drop("release_date_wiki", axis=1, inplace=True)
    
    
    # Handle Language - Drop Wikipedia
    # --------------------------------
    movies_df.drop("Language",axis=1,inplace=True)
    
    # Handle Production Companies - Drop Wikipedia
    # --------------------------------------------
    movies_df.drop("Production company(s)",axis=1,inplace=True)
    
    # Reorder the columns (and remove the 'video' column which contains only 
    # one value)
    # --------------------
    new_order = ['imdb_id','id','title_kaggle','original_title','tagline',
                 'belongs_to_collection','url','imdb_link',
                 'runtime','budget_kaggle','revenue','release_date_kaggle',
                 'popularity','vote_average','vote_count','genres',
                 'original_language','overview','spoken_languages','Country',
                 'production_companies','production_countries','Distributor',
                 'Producer(s)','Director','Starring','Cinematography',
                 'Editor(s)','Writer(s)','Composer(s)','Based on'
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
    
    try:
        load_movies_data(movies_df)
    except OperationalError as err: # Capture and log database error
        my_logger.error("Error loading movies table")
        # Log the database exception
        log_psycopg2_exception(err)
        # Rethrow the exception
        raise
        
    try:
        load_ratings_data()
    except OperationalError as err: # Capture and log database error
        my_logger.error("Error loading ratings table")
        # Log the database exception
        log_psycopg2_exception(err)
        # Rethrow the exception
        raise
    

# Load the wikipedia data, the kaggle movies metadata, and the kaggle ratings
# data into dataframes, then pass them to the Pipeline function to process 
# them and update the movies and ratings database tables accordingly.

if __name__ == "__main__":
    
    # Extract the Movielens data 
    try:
        movies_metadata = pd.read_csv(f'{data_dir}movies_metadata.csv',low_memory=False)
    except:
        e = sys.exc_info()[0]
        my_logger.critical(e, exc_info=True)
        sys.exit(1)
       
    # Extract the ratings data
    try:
        ratings = pd.read_csv(f'{data_dir}ratings.csv')
    except:
        e = sys.exc_info()[0]
        my_logger.critical(e, exc_info=True)
        sys.exit(1)
    
    # Extract the Wikipedia data
    try:
        with open(f'{data_dir}/wikipedia.movies.json', mode='r') as file:
            wiki_movies_raw = json.load(file)
    except:
        e = sys.exc_info()[0]
        my_logger.critical(e, exc_info=True)
        sys.exit(1)
        
        
    # Pass the dataframes to the Pipeline function for further processing
    try:
        Pipeline(wiki_movies_raw,movies_metadata,ratings)
    except:
        e = sys.exc_info()[0]
        my_logger.critical(e, exc_info=True)
        sys.exit(1)
