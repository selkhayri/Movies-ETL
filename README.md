# Movies-ETL

## Aim

The aim of this challenge is to create a processing pipeline that automates the processing of information from three different but related sourced and to deposit the processed information into a postgresql database.

## Extract

The data for this challenge is the following:

* Kaggle metadata (CSV)
* MovieLens ratings data (from Kaggle) (CSV)
* Wikipedia data (JSON)

## Transform

The following steps were taken during processing:

* Kaggle data was read into a dataframe
* Wikipedia data was read into a dataframe
* The two dataframes are merged

    * The columns containing similar information were compared and a strategy for retaining the most accurate data was devised
    * All the null or 0-values in the column that was retained were populated with the values from the column to be discarded
    * The other column was discarded

* The ratings dataframe is transformed into a pivot-table-like structure, grouping the movie id on the left and rating levels on the top. 

## Load

After the data was transformed, it was loaded into the movie_data database. 

<u>movies table</u>

In order to load the data into the movies table, several steps had to be taken to prepare the table to make sure that there would be no errors during the load. 

The following steps were taken:

* The contents of the movie table were wiped out.
* A comparison between the columns of movie table and the movies_df dataframe was undertaken. 
* If a discrepancy is found:

    * the columns from the movies table that are not present in the movies_df dataframe are dropped, and 
    * the columns from the movies_df that are not present in the movies table are added.

<i>Assumption 1</i>: <br/>
All the columns to be added will be text or varchar(30)

<u>ratings table</u>

The data from the ratings_df were inserted into the ratings table as follows:

* the data from the existing ratings table is wiped out
* the to_sql dataframe method is called for the ratings_df dataframe to insert the new rows into the ratings table
