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