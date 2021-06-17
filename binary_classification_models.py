import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def test_models_all_features(excel_file: str, dfs_name: str):
    """tests every model from generate_models
    and for both methods of normalisation

    Args:
        excel_files (List[str]): List of excel files with different methods of normalised features
        dfs_names:List[str]: list of names of normalisation methods

    Returns:
        List: List of accuracy scores with model name and which normalisation technique was used
    """
    models_and_names = generate_models()
    all_results = []
    df = pd.read_csv(excel_file)
    X = df[df.columns[10:]]
    Y = df["is higher lower"]
    split_num = round(0.9*len(X))  # 90/10 split on train / test
    X_train, X_test = X[0:split_num], X[split_num:]
    Y_train, Y_test = Y[0:split_num], Y[split_num:]
    all_results.append(run_models(X_train, X_test, Y_train, Y_test,
                                      models_and_names, dfs_name))
    return all_results


def generate_models():
    """generates full list of classifiers

    Returns:
        zipped list of classifier names and types
    """
    model_names = ["MLPClassifier", "AdaBoostClassifier", "SVC",
                   "KNeighborsClassifier", "GaussianProcessClassifier", "GaussianNB",
                   "QuadraticDiscriminantAnalysis", "DecisionTreeClassifier", "RandomForestClassifier",
                   "MLPClassifier"]
    models = [MLPClassifier(), AdaBoostClassifier(), SVC(),
              KNeighborsClassifier(), GaussianProcessClassifier(), GaussianNB(),
              QuadraticDiscriminantAnalysis(), DecisionTreeClassifier(), RandomForestClassifier()]
    models_and_names = zip(model_names, models)
    return models_and_names


def run_models(X_train, X_test, Y_train, Y_test, models_list, dataframe_name):
    """Runs each model from the zipped classifiers list,
    calculating their accuracy score

    Args:
        X_train 
        X_test 
        Y_train 
        Y_test 
        models_list : zipped models list returned by generate_models()
        dataframe_name : which normalisation method was used for the dataframe

    Returns:
        accuracy score and name of normalisation method and classifier type
    """
    results = []
    for type, model in models_list:
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_predict)*100
        results.append([type, accuracy, dataframe_name])
    return results


def get_lasso_weights_scores(excel_file: str, dfs_name, threshhold_func, func_arg2):
    """creates an array of indices of features to drop based on one of the two functions

    Args:
        excel_file (str):
        dfs_name ([type]): 
        threshhold_func ([type]): The function to be used to decide which features to drop, 
        func_arg2 ([type]): weighting_threshhold/ n features to keep depending on which
        threshhold_func is used
    """
    df = pd.read_csv(excel_file)
    X = df[df.columns[10:]]
    Y = df["is higher lower"]
    split_num = round(0.9*len(X))  
    X_train, X_test = X[0:split_num], X[split_num:]
    Y_train, Y_test = Y[0:split_num], Y[split_num:]
    LR = LogisticRegression(
        penalty='l1', solver='liblinear', max_iter=1000)
    LR.fit(X_train, Y_train)
    Y_predict = LR.predict(X_test)
    
    return(threshhold_func(LR.coef_[0],func_arg2))

#2 different functions for choosing features to drop

def find_low_weighted_features_by_threshhold(coefficients_array,weighting_threshhold):
    unwanted_indices = []
    for index, weighting in enumerate(coefficients_array):
        if abs(weighting) <= weighting_threshhold:
            unwanted_indices.append(index)
    return unwanted_indices

def find_strongest_weighted_features(coefficients_array,n):
    #finds n max weighted features indices to keep
    wanted_indices = np.argpartition(coefficients_array, -n)[-n:]
    #return all indices that aren't the max n weightings
    unwanted_indices = [i for i in range(len(coefficients_array)) if i not in wanted_indices]
    return unwanted_indices

def test_models_with_dropped_features(excel_file: str, dfs_name,threshhold_func,func_arg2):
    models_and_names = generate_models()
    all_results = []
    low_weight_indices_array = get_lasso_weights_scores(excel_file, dfs_name,threshhold_func,func_arg2)
    df = pd.read_csv(excel_file)
    X = df[df.columns[10:]]
    X = X.drop(X.columns[low_weight_indices_array], axis=1)
    Y = df["is higher lower"]
    split_num = round(0.9*len(X))  
    X_train, X_test = X[0:split_num], X[split_num:]
    Y_train, Y_test = Y[0:split_num], Y[split_num:]
    all_results.append(run_models(X_train, X_test, Y_train, Y_test,
                                      models_and_names, dfs_name))
    return all_results


if __name__ == "__main__":
    dfs_name = "Robust normalisation"
    print(test_models_with_dropped_features("preprocessed_price_data_minmax_features.csv",dfs_name,find_strongest_weighted_features,10))
    print(test_models_with_dropped_features("preprocessed_price_data_robust_features.csv",dfs_name,find_strongest_weighted_features,10))
    print(test_models_all_features("dollar_bar_minMax.csv",dfs_name))
    print(test_models_with_dropped_features("preprocessed_price_data_robust_features.csv",dfs_name,find_strongest_weighted_features,10))
    print(test_models_all_features("collinear_dropped_robust.csv",dfs_name))
    dfs_name = "MinMax normalisation"
    print(test_models_all_features("preprocessed_price_data_minmax_features.csv",dfs_name))
    print(test_models_with_dropped_features("preprocessed_price_data_minmax_features.csv",dfs_name,find_strongest_weighted_features,10))
    print(test_models_with_dropped_features("preprocessed_price_data_minmax_features.csv",dfs_name,find_low_weighted_features_by_threshhold,0))
    print(test_models_all_features("collinear_dropped_minmax.csv",dfs_name))
    print(test_models_all_features("preprocessed_price_data_minmax_features.csv",dfs_name))
