import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import testdata

testdata.data_to_csv()

MLP_test_df = pd.read_csv("put_testdata_5k.csv")
MLP_train_df = pd.read_csv('put_traindata_200k.csv')
MLP_validation_df = pd.read_csv('put_vdata_1k.csv')

# columns and labels
feature_columns = ["S", "K", "T", "r", "sigma"]
label_columns = ["value"]

# normalize the dataset
scaler = MinMaxScaler()

scaled_train_df = MLP_train_df.copy()
scaled_test_df = MLP_test_df.copy()

scaled_train_df[feature_columns] = scaler.fit_transform(scaled_train_df[feature_columns])
scaled_train_df[label_columns] = scaler.fit_transform(scaled_train_df[label_columns])

scaled_test_df[feature_columns] = scaler.fit_transform((scaled_test_df[feature_columns]))
scaled_test_df[label_columns] = scaler.fit_transform(scaled_test_df[label_columns])

# prepare data to be trained
x_train = np.array(scaled_train_df[feature_columns])
y_train = np.array(scaled_train_df[label_columns])

x_test = np.array(scaled_test_df[feature_columns])
y_test = np.array(scaled_test_df[label_columns])

# list of hyperparameters
HLS = (100, 50)
solver = 'adam'
learning_rate = ''
learning_rate_init = 0.0001

# parameter space
#parameter_space = {
#    'hidden_layer_sizes': [(64, 32), (100, 50), (100,), (50,), (50, 100), (8, 8, 8), (25, 50, 5), (5,), (25,)],
#    'activation': ['relu', 'identity'],
#    'solver': ['sgd', 'adam'],
#    'alpha': [0.0001, 0.05, 0.001, 0.002, 0.1],
#    'learning_rate': ['constant', 'adaptive'],
#}


# main function, MLPRegressor contained within
def main():
    mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 100), max_iter=1) #hidden_layer_sizes=(50, 100,), activation='relu', solver='adam', alpha= 0.001, learning_rate='adaptive', max_iter=500)

    mlp_reg.fit(X=x_train, y=y_train.ravel())

    predictions = mlp_reg.predict(x_test)

    # unscale predictions and test values
    import pdb;
    pdb.set_trace()
    predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_unscaled = scaler.inverse_transform(y_test)

    performance = mean_squared_error(y_test_unscaled, predictions_unscaled)

    print("Mean Squared Error = " + str(performance))
    pkl_filename = 'MalcolmSammon_Weights.pkl'
    #with open(pkl_filename, 'wb') as file:
    #    pickle.dump(mlp_reg, file)


    #print(str(mlp_reg.coefs_))

    #weights = pd.DataFrame(mlp_reg.coefs_)

    #weights.to_csv('MalcolmSammon_Weights500.csv')



    #clf = GridSearchCV(mlp_reg, parameter_space, n_jobs=-1, cv=3)
    #clf.fit(x_train, y_train.ravel())

    # Best parameter set
    #print('Best parameters found:\n', clf.best_params_)

    # All results
    #means = clf.cv_results_['mean_test_score']
    #stds = clf.cv_results_['std_test_score']
    #for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

if __name__ == "__main__":
    main()
