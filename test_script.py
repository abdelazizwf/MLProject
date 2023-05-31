import sys
import pickle

import pandas as pd

from sklearn import set_config
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def print_metrics(y_test, y_pred):
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the evaluation metrics and confusion matrix
    print(f"Accuracy: {round(accuracy * 100, 2)} %")
    print(f"Precision: {round(precision * 100, 2)} %")
    print(f"Recall: {round(recall * 100, 2)} %")
    print(f"F1 score: {round(f1 * 100, 2)} %")
    print(f"Confusion matrix:\n{conf_matrix}\n")


def run_models(names, data, target):
    path = "./saves/objects/"
    for name in names:
        with open(path + name + ".p", "rb") as f:
            model = pickle.load(f)

        predictions = model.predict(data)
        print(name)
        print_metrics(target, predictions)


if __name__ == "__main__":
    set_config(transform_output='pandas')

    if len(sys.argv) != 2:
        print('Usage: python test_script.py path_to_data')
        exit()

    test_path = sys.argv[1]

    final_columns = [
        'Unnamed: 0',
        'id',
        'owner_1_score',
        'years_in_business',
        'fsr',
        'INPUT_VALUE_ID_FOR_num_negative_days',
        'INPUT_VALUE_ID_FOR_num_deposits',
        'INPUT_VALUE_ID_FOR_monthly_gross',
        'INPUT_VALUE_ID_FOR_average_ledger',
        'INPUT_VALUE_ID_FOR_fc_margin',
        'INPUT_VALUE_ID_FOR_avg_net_deposits',
        'INPUT_VALUE_owner_4',
        'deal_application_thread_id'
    ]

    with open("./saves/objects/ord_columns.p", "rb") as f:
        ord_columns = pickle.load(f)

    with open("./saves/objects/bin_columns.p", "rb") as f:
        bin_columns = pickle.load(f)

    with open("./saves/objects/oh_columns.p", "rb") as f:
        oh_columns = pickle.load(f)

    with open("./saves/objects/num_columns.p", "rb") as f:
        num_columns = pickle.load(f)

    chosen_columns = ord_columns + bin_columns + oh_columns + num_columns

    data = pd.read_csv(test_path)

    target = data["completion_status"]

    data = data[data.columns.intersection(chosen_columns)]

    # Make all text lowercase
    filt = (data.dtypes == "object") | (data.dtypes == "bool")
    data.loc[:, filt] = data.loc[:, filt].applymap(
        str.lower, na_action='ignore')

    with open("./saves/objects/ord_pipeline.p", "rb") as f:
        ord_pipeline = pickle.load(f)

    index = data.columns.intersection(ord_columns)
    data[index] = ord_pipeline.transform(data[index])

    with open("./saves/objects/bin_pipeline.p", "rb") as f:
        bin_pipeline = pickle.load(f)

    index = data.columns.intersection(bin_columns)
    data[index] = bin_pipeline.transform(data[index])

    with open("./saves/objects/oh_pipeline.p", "rb") as f:
        oh_pipeline = pickle.load(f)

    index = data.columns.intersection(oh_columns)
    oh_data = oh_pipeline.transform(data[index])
    data.drop(columns=oh_columns, inplace=True)
    data = pd.concat([data, oh_data], axis=1)

    with open("./saves/objects/num_pipeline.p", "rb") as f:
        num_pipeline = pickle.load(f)

    index = data.columns.intersection(num_columns)
    data[index] = num_pipeline.transform(data[index])

    data = data[data.columns.intersection(final_columns)]

    models = [
        "adaboost_model", "bagging_model",
        "gradientboost_model", "random_forest_model"
    ]

    run_models(models, data, target)
