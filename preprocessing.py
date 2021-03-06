from datetime import date, timedelta
import pandas as pd
import numpy as np
from scipy.stats.stats import ttest_1samp
import ta
from scipy.stats import normaltest
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from datetime import datetime

def check_for_duplicates(excel_file_path):
    price_df = pd.read_csv(excel_file_path)
    timestamps = price_df['date']
    non_dupes = set()
    for date in timestamps:
        if date in non_dupes:
            return True
        non_dupes.add(date)
    return False

def create_higher_lower_array(excel_file_path: str, price_column_title:str, time_gap:int, output_file_path: str):
    """ Creates column in excel file holding whether the rise rose
    or fell in the next n (where n = time_gap) time window

    Args:
        String (excel_file): excel file with price data
        String (price_column_title): column header with price data
        int (time_gap): how many prices to jump before comparison
        String (output_file_path): csv file to be created
   
    """
    price_df = pd.read_csv(excel_file_path)
    closing_prices = price_df[price_column_title]
    bool_array = []
    for i in range(0,len(closing_prices)-1,time_gap):
        if closing_prices[i+1] > closing_prices[i]:
            bool_array.append(1)
        else:
            #classify's same price at close as class 0
            bool_array.append(0)
    bool_array.append(1)
    series = pd.Series(bool_array)
    price_df["is higher lower"] = series
    price_df.to_csv(output_file_path, index = False)

def reverse_columns(excel_file_path: str, output_file_path: str):
    """price data from some websites is given with latest price at the top, 
    reverses it to have latest price last

    Args:
       String (excel_file): excel file with price data
       String (output_file_path): csv file to be created
    """
    price_df = pd.read_csv(excel_file_path)
    for column in price_df:
        price_df[column] = price_df[column].values[::-1]
    price_df.to_csv(output_file_path)

def create_all_ta_features(excel_file_path: str, output_file_path: str):
    """ Uses ta library to generate all all possible indicator values to use as features

    Args:
        excel_file_path (str): input excel file 
        output_file_path (str): output excel file
    """
    price_df = pd.read_csv(excel_file_path)
    price_df = ta.add_all_ta_features(price_df, open= "open", high = "high", low = "low", close = "close", volume = "Volume BTC", fillna = True)
    price_df.to_csv(output_file_path,index=False)

def test_feature_normality(feature_column):
    """ tests each feature column for normality using normaltest() which uses D'Agostino's K^2 test.

    Args:
        feature_column : column of data for feature

    Returns:
        True if normally distributed, else False
    """
    statistics, p = normaltest(feature_column)
    alpha = 0.05
    if p > alpha:
        return True 
    else:
        return False 

def test_total_features_normality(excel_file_path: str):
    """tests all features generated by ta library for normal distribution,
    useful for determining whether to normalise or standardise

    Args:
        excel_file_path (str): input excel file

    Returns:
        decimal value between 0 and 1 representing what percentage of the features
        are normally distributed
    """
    price_df = pd.read_csv(excel_file_path)
    features = price_df[price_df.columns[10:]]
    total_normal = 0
    total_not_normal = 0
    for column in features:
        if test_feature_normality(features[column].to_list()):
            total_normal+= 1
        else:
            total_not_normal +=1
    return (total_normal / (total_normal + total_not_normal))

def create_robust_normalised_csv(excel_file_path: str, output_file_path: str):
    """uses RobustScaler to create a normalised feature csv

    Args:
        excel_file_path (str): input file 
        output_file_path (str): output file 
    """
    price_df = pd.read_csv(excel_file_path)
    for column in price_df[price_df.columns[10:]]:
       price_df[column] = RobustScaler().fit_transform(price_df[[column]])
    price_df.to_csv(output_file_path,index=False)

def create_min_max_csv(excel_file_path: str, output_file_path: str):
    """uses MinMaxScaler to create a normalised feature csv

    Args:
        excel_file_path (str): input file 
        output_file_path (str): output file 
    """
    price_df = pd.read_csv(excel_file_path)
    for column in price_df[price_df.columns[10:]]:
       price_df[column] = MinMaxScaler().fit_transform(price_df[[column]])
    price_df.to_csv(output_file_path,index=False)

def fix_bad_data(excel_file_path: str, output_file_path: str):
    """Some bitcoin data csv's turned out to randomly swap the BTC volume and USD volume
    values halfway through the column, this corrects the issue

    Args:
        excel_file_path (str): [description]
        output_file_path (str): [description]
    """
    price_df = pd.read_csv(excel_file_path)
    USD = price_df['Volume USD']
    BTC = price_df['Volume BTC']
    for index, value in enumerate(USD):
        if BTC[index] > USD[index]:
            temp = USD[index]
            USD[index] = BTC[index]
            BTC[index] = temp
    price_df['Volume USD'] = USD
    price_df['Volume BTC'] = BTC
    price_df.to_csv(output_file_path,index=False)
            

#calculate_vif found at:
#https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-pandas-columns-in-python

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def drop_collinear_features(excel_file_path: str, output_file_path: str):
    #creates new excel file that keeps only features that are not collinear
    price_df = pd.read_csv(excel_file_path)
    X = price_df[price_df.columns[10:]]
    df = price_df[price_df.columns[0:10]]
    dropped_X = calculate_vif_(X)
    df = pd.concat([df,dropped_X],axis=1)
    df.to_csv(output_file_path,index=False)

def create_dict_from_df(excel_file_path):
    """takes a price_df and creates a dict for use in create_dollar_bar_df

    Args:
        excel_file_path

    Returns:
        dictionary with correctly parsed dates
    """
    price_df = pd.read_csv(excel_file_path)
    date_column = price_df['date']
    for index, date in enumerate(date_column):
        date_column[index] = datetime.strptime(date,"%d/%m/%Y %H:%M")
    price_df['date'] = date_column
    price_dict = price_df.to_dict('records')
    return price_dict

def create_dollar_bar_df(excel_file_path, threshhold,output_file_path,time_delta_hours,time_delta_minutes):
    """creates a dollar bar df where the timestamps are switched for when the 
    threshhold dollar amount is sold. Essentially the new timestamps, instead of being
    linearly separated by 1 hour, are now separated by how long it took for the threshold amount
    of dollars of to be traded.

    Args:
        excel_file_path 
        threshhold (int): the threshold for which after that many dollars are sold, 
        a new dollar bar is created
        output_file_path 
        time_delta_hours(int) : should be 1 if using data every hour, 0 if using data every minute
        time_delta_minutes(int): should be 0 if using data every hour, 1 if using data every minute
    """
    price_dict = create_dict_from_df(excel_file_path)
    bars = []
    current_dollar_volume = 0
    high, low = 0,100000000000000000000
    for i in range(len(price_dict)):
        next_unix,next_symbol,next_volume_USD,next_open, next_high, next_low, next_close, next_date, next_volume_BTC = [price_dict[i][j] for j in ['unix','symbol','Volume USD','open','high','low','close','date','Volume BTC']]
        average_price = (next_open + next_close)/2
        dollar_volume = next_volume_BTC * average_price
        high, low = max(high, next_high), min(low, next_low)
        if dollar_volume + current_dollar_volume >= threshhold:
            bar_date = next_date + timedelta(hours=time_delta_hours,minutes = time_delta_minutes)
            bars += [{'unix': next_unix, 'date':bar_date,'symbol':next_symbol,'open':next_open,'high':high,'low':low,'close':next_close,'Volume USD':next_volume_USD,'Volume BTC':next_volume_BTC}]
            current_dollar_volume = 0
            high, low = 0,100000000000000000000
        else:
            current_dollar_volume += dollar_volume
    price_df = pd.DataFrame(bars)
    price_df.to_csv(output_file_path,index=False)

def ta_for_dollar_bars(excel_file_path, output_file_path):
    """generates selected ta features for dollar bar type csv file. Note any indicator
    that uses volume is not included, as one of the properties of dollar bars vs time bars is 
    volume is inherently included in the bar already.

    Args:
        excel_file_path 
        output_file_path
    Returns:
        updated csv with selected features calculated 
    """
    price_df = pd.read_csv(excel_file_path)
    awesomeOscillator = ta.momentum.AwesomeOscillatorIndicator(high = price_df['high'],low = price_df['high'],fillna = True)
    ao = awesomeOscillator.awesome_oscillator()
    percentageOscillator = ta.momentum.PercentagePriceOscillator(close = price_df['close'],fillna = True)
    po = percentageOscillator.ppo()
    momentumStochRSI = ta.momentum.StochRSIIndicator(close = price_df['close'],fillna = True)
    stoch = momentumStochRSI.stochrsi()
    ultimateOscillator = ta.momentum.UltimateOscillator(high = price_df['high'],low = price_df['high'],fillna = True,close = price_df['close'])
    uo = ultimateOscillator.ultimate_oscillator() 
    avgTrueRange = ta.volatility.AverageTrueRange(high = price_df['high'],low = price_df['high'],fillna = True,close = price_df['close'])
    atr = avgTrueRange.average_true_range()
    mAVG = ta.volatility.bollinger_mavg(close = price_df['close'],fillna = True)
    ADX = ta.trend.ADXIndicator(high = price_df['high'],low = price_df['high'],fillna = True,close = price_df['close'])
    adx = ADX.adx()
    MACD = ta.trend.MACD(close = price_df['close'],fillna = True)
    macd = MACD.macd()
    weightedAVG = ta.trend.WMAIndicator(close = price_df['close'],fillna = True)
    wavg = weightedAVG.wma()
    price_df['awesomeOscillator'] = ao
    price_df['percentageOscillator'] = po
    price_df['momentumStochRSI'] = stoch
    price_df['ultimateOscillator'] =  uo
    price_df['avgTrueRange'] = atr
    price_df['mAVG'] = mAVG
    price_df['ADX'] = adx
    price_df['MACD'] = macd
    price_df['weightedAVG'] = wavg
    price_df.to_csv(output_file_path,index=False)

def add_current_close_as_feature(excel_file_path, output_file_path):
    """used for models where you are predicting price rather than binary
    classification, inserts next closing price as target / Y_actual

    Args:
        excel_file_path 
        output_file_path
    """
    price_df = pd.read_csv(excel_file_path)
    close = price_df['close']
    future_close = []
    for i in range(len(close)-1):
        future_close.append(close[i+1])
    future_close.append(future_close[-1]) # fills in final 'next close' with duplicate price since that is end of data
    series = pd.Series(future_close)
    price_df["next close"] = series
    price_df = price_df.drop(columns = ["is higher lower"])
    price_df.to_csv(output_file_path, index = False)

def add_close_price_as_feature(excel_file_path, output_file_path):
    price_df = pd.read_csv(excel_file_path)
    current_close = price_df['close']
    copy = []
    for price in current_close:
        copy.append(price)
    price_df['close_copy'] = copy
    price_df.to_csv(output_file_path,index = False)

if __name__ == "__main__":
    fix_bad_data("preprocessed_price_data.csv", "fixed_price_data.csv")
    create_higher_lower_array("preprocessed_price_data.csv", "close", 1, "preprocessed_price_data.csv")
    create_all_ta_features("fixed_price_data.csv", "preprocessed_price_data_all_features.csv")
    print(test_total_features_normality("preprocessed_price_data_all_features.csv"))
    create_robust_normalised_csv("preprocessed_price_data_all_features.csv", "preprocessed_price_data_robust_features.csv")
    create_min_max_csv("preprocessed_price_data_all_features.csv", "preprocessed_price_data_minmax_features.csv")
    drop_collinear_features("preprocessed_price_data_all_features.csv","price_data_collinear_dropped.csv")
    create_robust_normalised_csv("price_data_collinear_dropped.csv", "collinear_dropped_robust.csv")
    create_min_max_csv("price_data_collinear_dropped.csv", "collinear_dropped_minMax.csv")
    create_dollar_bar_df("fixed_price_data.csv",10000000,"dollar_bar_data.csv",time_delta_hours=1,time_delta_minutes=0)
    create_higher_lower_array("dollar_bar_data.csv", "close", 1, "dollar_bar_data.csv")
    ta_for_dollar_bars("dollar_bar_data.csv","dollar_bar_data_with_indicators.csv")
    create_min_max_csv("dollar_bar_data_with_indicators.csv", "dollar_bar_minMax")
    add_current_close_as_feature("fixed_price_data.csv","preprocessed_non_binary.csv")
    create_all_ta_features("preprocessed_non_binary.csv", "non_binary_all_features.csv")
    drop_collinear_features("non_binary_all_features.csv","non_binary_all_features.csv")
    create_min_max_csv("non_binary_all_features.csv", "non_binary_all_features_minmax.csv")
    create_min_max_csv("dollar_bar_data_with_indicators.csv", "dollar_bar_minMax.csv")
    add_current_close_as_feature("fixed_price_data.csv","preprocessed_non_binary.csv")
    ta_for_dollar_bars("non_binary_dollar.csv","non_binary_dollar.csv")
    add_close_price_as_feature("non_binary_dollar.csv","non_binary_dollar.csv")
