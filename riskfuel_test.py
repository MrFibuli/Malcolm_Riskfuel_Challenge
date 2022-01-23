import pandas as pd
import argparse
import sys

# Load whatever imports you need, but make sure to add them to the requirements.txt file.

# My imports
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def riskfuel_test(df: pd.DataFrame) -> float:
    """
    Riskfuel Testing Function
    by <team-name>: <member_1> <member_2> .... <member_k>

    arguments: pandas DataFrame type with the following columns.. ['S','K','T','r','sigma','value'] all are of type float32
    ouputs: mean absolute error (float)

    Once you have finished model training/developemnt you must save the model within the repo and load it in using this function.

    You are free to import any python packages you desire but you must add them to the requirements.txt file.

    This function must do the following:
        - Successfully load your own model.
        - Take in a dataframe consisting of (N x 6) float32's.
        - Take the (N x 5) columns regarding the inputs to the pricer ['S','K','T','r','sigma'] and have your model price them.
        - Return the Mean  Absolute Error of the model.

    Do not put the analytic pricer as part of your network.
    Do not do any trickery with column switching as part of your answer.

    These will be checked by hand, any gaslighting will result in automatic disqualification.

    The following example has been made available to you.
    """

    # TEAM DEFINITIONS.
    team_name = "The Jammin' Sammons"  # adjust this
    members = ["Malcolm Sammon"]  # adjust these

    print(f"\n\n ============ Evaluating Team: {team_name} ========================= ")
    print(" Members :")
    for member in members:
        print(f" {member}")
    print(" ================================================================ \n")

    # ===============   Example Code  ===============

    # My model uses PyTorch but you can use whatever package you like,
    # as long you write code to load it and effectively calculate the mean absolute aggregate error.

    # LOAD MODEL
    with open('MalcolmSammon_Weights.pkl', 'rb') as file:
        pickle_model = pickle.load(file)


    # EVALUATE MODEL

    # Acquire inputs/outputs
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_test = np.array(df[["S", "K", "T", "r", "sigma"]])
    y_test = np.array((df[["value"]].to_numpy()))

    x_test = scaler_x.fit_transform(x_test)
    y_test_gsags = scaler_y.fit_transform(y_test)

    # Pass data through model
    predictions = pickle_model.predict(x_test)
    #import pdb; pdb.set_trace()
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1))

    # Calculate mean squared error
    result = mean_squared_error(y_test, predictions)

    # Return performance metric; must be of type float
    return result


# A SIMPLE MODEL.
class MLPRegressor(MLPRegressor):
    """
    Example of a Neural Network that could be trained price a put option.
    """
    #mlp_reg = MLPRegressor(hidden_layer_sizes=(50, 100), max_iter=500)




def get_parser():
    """Parses the command line for the dataframe file name"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame_name", type=str)
    return parser


def main(args):
    """Parses arguments and evaluates model performance"""

    # Parse arguments.
    parser = get_parser()
    args = parser.parse_args(args)

    # Load DataFrame and pass through riskfuel_test function.
    df = pd.read_csv(args.data_frame_name)
    performance_metric = riskfuel_test(df)

    # Must pass this assertion
    assert isinstance(performance_metric, float)

    print(f" MODEL PERFORMANCE: {performance_metric} \n\n")


if __name__ == "__main__":
    main(sys.argv[1:])