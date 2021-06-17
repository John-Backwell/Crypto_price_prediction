from textwrap import indent
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import kernel_ridge
from sklearn import tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from binary_classification_models import get_lasso_weights_scores
from binary_classification_models import find_strongest_weighted_features

def generate_models():
    #same format as in binary classifcation but different models used
    model_names = ["KNeighborsRegressor","LinearRegression","Lasso",
    "BayesRidge","KernalRidge","SGD","DecisionTree"]
    models = [neighbors.KNeighborsRegressor(),linear_model.LinearRegression(),
    linear_model.Lasso(alpha = 0.1,max_iter=100000),linear_model.BayesianRidge(),
    kernel_ridge.KernelRidge(alpha=0.1),
    linear_model.SGDRegressor(max_iter=100000),tree.DecisionTreeRegressor()]
    models_and_names = zip(model_names, models)
    return models_and_names

def run_models(X_train, X_test, Y_train, Y_test, models_list, dataframe_name):
    results = []
    
    for type, model in models_list:
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
        mse = mean_squared_error(Y_test, Y_predict,squared= False)
        results.append([type, mse, dataframe_name])
        plot_price_predictions(Y_predict, Y_test,type,type)
        title = type + " focused"
        plot_price_predictions(Y_predict[-100:],Y_test[-100:],type,title)
    return results

def test_models_all_features(excel_file: str, dfs_name: str):
    models_and_names = generate_models()
    all_results = []
    df = pd.read_csv(excel_file)
    X = df[df.columns[10:]]
    Y = df["next close"]
    split_num = round(0.9*len(X))  # 90/10 split on train / test
    X_train, X_test = X[0:split_num], X[split_num:]
    Y_train, Y_test = Y[0:split_num], Y[split_num:]
    all_results.append(run_models(X_train, X_test, Y_train, Y_test,
                                      models_and_names, dfs_name))
    return all_results

def plot_price_predictions(Y_predict, Y_true,title, saveName):
    x = [i for i in range(len(Y_predict))]
    plt.plot(x,Y_predict,'r-',label = 'Predicted Price')
    plt.plot(x,Y_true,'b-',label = 'Actual Price')
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title = title
    plt.legend(loc="upper left")
    plt.savefig(saveName)
    plt.close()
    

if __name__ == "__main__":
    dfs_name = "MinMax normalisation"
    print(test_models_all_features("non_binary_all_features_minmax.csv",dfs_name))
    
