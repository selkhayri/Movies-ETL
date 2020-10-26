# Movies-ETL

## Aim

The aim of this project is to create a processing pipeline that automates the processing of information from three different but related sources and to deposit the processed information into a PostgreSQL database.

## Extract

The data for this project is the following:

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

* The ratings dataframe is transformed into a pivot-table-like structure, grouping the movie id on the left and rating levels on the top, named ```rating_counts```.
* The movies_with_ratings_df dataframe is created by merging the transfromed dataframes, ```movies_df``` and ```rating_counts```, using a left-join on the column ```kaggle_id```

## Load

After the data was transformed, it was loaded into the movie_data database.

#### movies table

In order to load the data into the movies table, several steps had to be taken to prepare the table to make sure that there would be no errors during the load.

The following steps were taken:

* The contents of the movie table were wiped out.
* A comparison between the columns of movie table and the ```movies_df``` dataframe was undertaken. 
* If a discrepancy is found:

    * the columns from the movies table that are not present in the ```movies_df``` dataframe are dropped

    * the columns from the ```movies_df``` that are not present in the movies table are added.

#### ratings table

The data from the ratings_df were inserted into the ratings table as follows:

* the data from the existing ratings table is wiped out,
* the ```to_sql``` dataframe method is called for the ```ratings_df``` dataframe to insert the new rows into the ```ratings``` table

## Error Handling

```try-except``` blocks were used for the following operations:

* Table load operations
* File access operations
* The call to the Pipeline function

In other places, ```exceptions``` were ```raise```d if the following conditions occurred:

* Function calls with invalid parameters
* Casting variables to different data types

## Logging

Since all the messages that were sent to the console had to be removed from the pipeline, logging was enabled so that, in the case of faulty behaviour, log files can be used for diagnosis and troubleshooting.

## Assumptions

1. Any columns that need to be added to the movies table are of type ```text/varchar(30)```.

2. The data files are retrieved manually and placed in the data folder before the pipeline is run.

3. The administrator of the pipeline has the appropriate credentials and permissions to retrieve the required data files.

4. There will not be any alterations to the structures of the ratings data and the movies data which would necessitate the revision of the processing in the pipeline.

5. The type of date in the ratings table is constant and, aside from wiping the table clean before each insert, no other alterations will be required. There will be no need, as in the case of the movies table, to drop columns or add columns dynamically.
