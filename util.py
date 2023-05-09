from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def drop_empty_features(data, cutoff=0.5):
    """Drop features with NaN values more that cutoff."""
    n_rows = data.shape[0]
    empty_features = []
    for column in data.columns:
        na_values = sum(data[column].isna())
        ratio = na_values / n_rows
        if ratio > cutoff:
            empty_features.append(column)
    new_data = data.drop(labels=empty_features, axis=1)
    return new_data


def get_numerical_features(data):
    """Return DataFrame with the numerical features."""
    return data.select_dtypes(["int64", "float64"])


def get_categorical_features(data):
    """Return DataFrame with the categorical features."""
    return data.select_dtypes(["object", "bool"])


def visualize_distributions(data, label_size=4):
    """Display the histograms of all features."""
    fig = data.hist(xlabelsize=label_size, ylabelsize=label_size)
    for x in fig.ravel():
        x.title.set_size(label_size)
    return


def print_na_ratio(data):
    """Print the NaN ratio of every feature."""
    rows_no = data.shape[0]
    width = max([len(c) for c in data.columns])
    for column in data.columns:
        na_values = sum(data[column].isna())
        ratio = round((na_values / rows_no) * 100, 2)
        print(
            "{column:<{width}} {ratio}%".format(
                column=column,
                ratio=ratio,
                width=width
            )
        )
    return


def print_categories_ratio(data):
    """Print the ratio of categories in every feature."""
    for column in data.columns:
        counter = Counter(data[column])
        t = counter.total()
        ratios = [
            (l, round((n/t)*100, 2)) for l, n in counter.most_common()
        ]
        print(column + "\n", *ratios, "\n")
    return



