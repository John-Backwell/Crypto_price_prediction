from math import comb
from sys import exc_info
import numpy as np
from numpy.core.numeric import count_nonzero
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from binary_classification_models import get_lasso_weights_scores
from binary_classification_models import find_strongest_weighted_features
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from regression_models import plot_price_predictions
from sklearn import tree

def calculate_combined_model_accuracy(excel_file_1, excel_file_2, model_1,model_2,df_name,threshhold_func,func_arg2):
    """Combines a binary classification model with a regression model, and creates a combined decisions array
    that only holds values for where the models agreed: IE the binary model classified it as a predicted price rise, and
    the regression model predicted a price higher than the current closing price.

    Args:
        excel_file_1 
        excel_file_2 
        model_1 
        model_2 
        df_name 
        threshhold_func 
        func_arg2 

    Returns:
        combined decisions array, accuracy of combined decisions, what percentage of total opportunities for trading where taken
        by the combined model, actual values on whether price rose or fell.
    """
    indices = get_lasso_weights_scores(excel_file_1,df_name,threshhold_func,func_arg2)
    df_1 = pd.read_csv(excel_file_1)
    df_2 = pd.read_csv(excel_file_2)
    X_1 = df_1[df_1.columns[10:]]
    X_1 = X_1.drop(X_1.columns[indices], axis=1)
    Y_1 = df_1["is higher lower"]
    X_2 = df_2[df_2.columns[10:]]
    Y_2 = df_2["next close"]
    split_num = round(0.9*len(X_1))
    X_train_1, X_test_1 = X_1[0:split_num], X_1[split_num:]
    Y_train_1, Y_test_1 = Y_1[0:split_num], Y_1[split_num:]
    model_1.fit(X_train_1,Y_train_1)
    Y_predict_1 = model_1.predict(X_test_1)
    #print(accuracy_score(Y_test_1, Y_predict_1)*100)
    X_train_2, X_test_2 = X_2[0:split_num], X_2[split_num:]
    Y_train_2, Y_test_2 = Y_2[0:split_num], Y_2[split_num:]
    model_2.fit(X_train_2,Y_train_2)
    Y_predict_2 = model_2.predict(X_test_2)
    #print(mean_squared_error(Y_test_2, Y_predict_2,squared= False))
    combined_decisions = []
    close = df_2['close']
    close = close[split_num:].tolist()
    for index, value in enumerate(Y_predict_1):
        if Y_predict_1[index] == 1 and Y_predict_2[index]>= close[index]:
            combined_decisions.append((index,1))
        elif Y_predict_1[index] == 0 and Y_predict_2[index]<= close[index]:
            combined_decisions.append((index,0))
    return combined_decisions, calculate_accuracy(combined_decisions,Y_test_1.tolist()), (len(combined_decisions)/len(Y_test_1)*100),Y_predict_1,Y_predict_2

def calculate_accuracy(combined_decisions, Y_test_1):
    """calculates percentage accuracy of a combined decisions array

    Args:
        combined_decisions: predicted values for combined model
        Y_test_1: True values
    Returns:
       percentage accuracy of predictions
    """
    total_decisions = len(combined_decisions)
    correct_decisions = 0
    for index, decision in combined_decisions:
        if decision == Y_test_1[index]:
            correct_decisions +=1
    return correct_decisions / total_decisions * 100


def simulate_trading_period_naive(decisions, excel_file, starting_cash, starting_BTC):
    """Comparing the percentage profitability of a naive trading strategy (in this case buying and
    selling 1 BTC at a time) against just holding the starting BTC and cash throughout the test period.

    Args:
        decisions: decisions array
        excel_file 
        starting_cash: how much cash the model starts with
        starting_BTC: how much btc the model starts with
    Returns:
        the percentage profit of the model compared to holding, how much BTC it ends test period with, how much cash
    """
    running_BTC = starting_BTC
    running_cash = starting_cash
    price_df = pd.read_csv(excel_file)
    split_num = round(0.9*len(price_df))
    close = price_df['close']
    close = close[split_num:].tolist()
    for index, value in decisions:
        if value == 1:
            running_BTC, running_cash = buy_BTC(running_BTC,running_cash,close[index],1)
        elif value == 0:
            running_BTC, running_cash = sell_BTC(running_BTC,running_cash,close[index],1)

    percentage_profit = (((running_BTC * close[-1]) + running_cash) - ((starting_BTC * close[-1]) + starting_cash))/((starting_BTC * close[-1]) + starting_cash)
    return percentage_profit * 100, running_BTC,running_cash

def buy_BTC(current_BTC, current_cash, price,num):
    if current_cash >= price * num:
        current_BTC = current_BTC + num
        current_cash = current_cash - (num *price)
    return current_BTC,current_cash

def sell_BTC(current_BTC, current_cash, price,num):
    if current_BTC >= num:
        current_BTC = current_BTC - num
        current_cash = current_cash + (num*price)
    return current_BTC, current_cash




if __name__ == "__main__":
    results = calculate_combined_model_accuracy("preprocessed_price_data_robust_features.csv","non_binary_all_features_minmax.csv",RandomForestClassifier(),tree.DecisionTreeRegressor(),"minMax",find_strongest_weighted_features,10)
    print(results[1])
    print(results[24])
    decisions = results[0]
    print(simulate_trading_period_naive(decisions,"preprocessed_price_data_robust_features.csv",10000000,10))
    single_model_decisions = enumerate(results[-2])
    regressor_predictions = enumerate(results[-1])
    price_df = price_df = pd.read_csv("non_binary_all_features_minmax.csv")
    close = price_df['close']
    split_num = round(0.9*len(price_df))
    close = close[split_num:].tolist()
    regressor_decisions = []
    for index, value in regressor_predictions:
        if value >= close[index]:
            regressor_decisions.append((index, 1))
        else:
            regressor_decisions.append((index,0))
    print(simulate_trading_period_naive(single_model_decisions,"preprocessed_price_data_robust_features.csv",10000000,10))
    print(simulate_trading_period_naive(regressor_decisions,"non_binary_all_features_minmax.csv",10000000,10))

