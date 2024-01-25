import pandas as pd

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