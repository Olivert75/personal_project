#Import for data manipulation
import pandas as pd
from sklearn.model_selection import train_test_split
import os
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import re 


def get_data(csv_file):
    '''
    This function will take in any csv file and return it as a dataframe
    '''
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
    else:
        print('The file does not exist in current directory')
    return df

def clean_data(df):
    '''
    This function will drop some columns that arent needed for exploration progress
    '''
    df = df.drop(columns = ['destroyedTopInhibitor','destroyedMidInhibitor','destroyedBotInhibitor',
                            'lostTopInhibitor','lostMidInhibitor','lostBotInhibitor',
                            'destroyedTopNexusTurret','destroyedMidNexusTurret','destroyedBotNexusTurret',
                            'lostTopNexusTurret','lostMidNexusTurret','lostBotNexusTurret','destroyedTopBaseTurret',
                            'destroyedMidBaseTurret','destroyedBotBaseTurret','lostTopBaseTurret','lostMidBaseTurret',
                            'lostBotBaseTurret','destroyedTopInnerTurret','destroyedMidInnerTurret','destroyedBotInnerTurret',
                            'lostTopInnerTurret','lostMidInnerTurret','lostBotInnerTurret',
                            'destroyedTopOuterTurret','destroyedMidOuterTurret',
                            'destroyedBotOuterTurret','lostTopOuterTurret','lostMidOuterTurret','lostBotOuterTurret'])
    return df

def convert (camel_input):
    '''
    This function will rename string values or columns
    '''
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))


def split_data(df,target):
    '''
    This function takes in a dataframe and a target variable and split the data into 3: train, validate and test
    Establish train+validate set 80% of original data and then repeat the process 
    Split train+validate  into train, validate separately 
    '''
    train_validate, test = train_test_split(df, 
                                            test_size=.2, 
                                            random_state=123, 
                                            stratify=df[target])

    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate[target])
    return train, validate, test

def wrangle_data(target, csv_file):
    '''
    This function will use 3 functions above and return train, validate and test 
    '''
    df = get_data(csv_file)
    df = clean_data(df)
    train, validate, test = split_data(df, target)

    return df, train, validate, test

def standard_scale_data(X_train, X_validate, X_test):
    """
    Takes in X_train, X_validate and X_test dfs with numeric values only
    Returns scaler, X_train_scaled, X_validate_scaled, X_test_scaled dfs
    """
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index = X_train.index, columns = X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index = X_validate.index, columns = X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled

def split_X_y(train, validate, test, target):
    '''
    Splits train, validate, and test into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

