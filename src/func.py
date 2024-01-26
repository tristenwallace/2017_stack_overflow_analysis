import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



def get_description(col_name, schema):
    '''
    INPUT:
        - schema (schema): pandas dataframe with the schema of the data
        - col_name (string): name of the column you want a description for
    
    OUTPUT:
        - desc (string): description of the column
    '''
    
    desc = schema[schema["Column"] == col_name]['Question']
    
    return desc



def separate_values(df, col):
    ''' 
    INPUT:
        df - the pandas dataframe you want to search
        col - the column name you want to look through. Column values are 
                strings separated by "; "
    
    OUTPUT:
        new_df - a dataframe of each unique string from column with the
                    count of how often it shows up
    '''
    temp_dict = {}
    for i in df.iterrows():
        for text in df[col][i[0]].split(";"):
            if text.strip() in temp_dict.keys():
                temp_dict[text.strip()] += 1
            else:
                temp_dict[text.strip()] = 1
    new_df = pd.DataFrame.from_dict(temp_dict, orient="index").reset_index()
    new_df.rename(columns={"index":col, 0:"Count"}, inplace=True)
    
    return new_df


def get_df_props(df):
    df = df.CousinEducation.value_counts().reset_index()
    df.rename(columns={"CousinEducation":"Method"}, inplace=True)
    df = separate_values(df, "Method")
    df["Prop"] = df["Count"]/df["Count"].sum()
    df.drop("Count", axis=1, inplace=True)
    
    
    return df



def create_dummy_df(df, cat_cols, dummy_na):
    '''
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in  cat_cols:
        try:
            # for each cat add dummy var, drop original column
            df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)], axis=1)
        except:
            continue
    return df



def clean_fit_linear_mod(df, target_col, cat_cols, dummy_na, test_size=.3, rand_state=42):
    '''
    INPUT:
        df - a dataframe holding all the variables of interest
        response_col - a string holding the name of the column 
        cat_cols - list of strings that are associated with names of the categorical columns
        dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
        test_size - a float between [0,1] about what proportion of data should be in the test dataset
        rand_state - an int that is provided as the random state for splitting the data into training and test 
    
    OUTPUT:
        test_score - float - r2 score on the test data
        train_score - float - r2 score on the test data
        lm_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    #Drop the rows with missing response values
    df  = df.dropna(subset=[target_col], axis=0)

    #Drop columns with all NaN values
    df = df.dropna(how='all', axis=1)

    #Dummy categorical variables
    df = create_dummy_df(df, cat_cols, dummy_na)

    # Mean function
    fill_mean = lambda col: col.fillna(col.mean())
    # Fill the mean
    df = df.apply(fill_mean, axis=0)

    # Train test split
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    # Fit Model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Scores
    test_score = reg.score(X_test, y_test)
    train_score = reg.score(X_train, y_train)

    return test_score, train_score, reg, X_train, X_test, y_train, y_test