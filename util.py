from collections import Counter

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


def print_categories_ratio(data):
    """Print the ratio of categories in every feature."""
    for column in data.columns:
        counter = Counter(data[column])
        t = counter.total()
        ratios = [
            (l, round((n/t)*100, 2)) for l, n in counter.most_common()
        ]
        print(column + "\n", *ratios, "\n")


def visualize_outliers(data, columns, quantile_cutoffs=0.01):
    """Display column plots before and after removing outliers."""
    if type(columns) == str:
        columns = list(columns.split())

    num_cols = len(columns)
    if type(quantile_cutoffs) == float:
        quantile_cutoffs = [quantile_cutoffs] * num_cols

    assert len(quantile_cutoffs) == num_cols

    fig, axes = plt.subplots(nrows=num_cols, ncols=2, squeeze=False)

    for i, column in enumerate(columns):
        q_high = data[column].quantile(1 - quantile_cutoffs[i])
        q_low = data[column].quantile(quantile_cutoffs[i])

        data[column].plot(ax=axes[i, 0])

        filt = (data[column] < q_high) & (data[column] > q_low)
        data.loc[filt, column].plot(ax=axes[i, 1])


def visualize_rfe_scores(rfecv):
    """Visualize the mean accuracy scores over the number of features."""
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.plot(
        range(1, n_scores + 1),
        rfecv.cv_results_["mean_test_score"],
    )
    plt.title("Recursive Feature Elimination")
