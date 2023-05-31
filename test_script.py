import sys
import pickle

import pandas as pd

from sklearn import set_config

from util import print_metrics


def run_saved_models(names, data, target):
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

    with open("./saves/objects/final_columns.p", "rb") as f:
        final_columns = pickle.load(f)

    data = data[data.columns.intersection(final_columns)]

    models = [
        "adaboost_model", "bagging_model",
        "gradientboost_model", "random_forest_model"
    ]

    run_saved_models(models, data, target)
