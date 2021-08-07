import numpy as np
import pandas as pd
import sys

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the dataset which includes the messages and categories
    
    Arguments:
    - The file path of the messages csv file
    - The file path of the categories csv file
    
    Returns:
    - The inner merged dataset using the "id" column for joining the 2 tables
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    data = pd.merge(messages, categories, on='id', how='inner')
    
    return data


def clean_data(df):
    '''
    Cleans the data
    
    Arguments:
    - The unclean dataset (output of the load_data function)
    
    Returns:
    - The clean dataset
    '''
    # extract categories column for processing/cleaning
    categories = df.categories.str.split(pat=';', expand=True)
    
    # rename categories columns (remove the dash and number)
    first_row = df.iloc[0, :]
    col_names = first_row.apply(lambda x:x[:-2])
    categories.columns = col_names
    
    # now use the number that was after the dash (in previous step) as cell value
    for column in catogeries:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    # replace categories column with clean one then drop duplicates
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(subset='id', inplace=True)
    
    return df


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()