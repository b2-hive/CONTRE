import os
import json
import root_pandas


def validation_import(parameter_file):
    """Return pd.DataFrame of weights and list with test and train samples."""

    # open resultfile
    with open(parameter_file) as f:
        parameters = json.load(f)

    name = parameters.get("name")
    result_path = parameters.get("result_path")
    result_file = os.path.join(
        result_path,
        "name="+name,
        "validation_results.json")

    with open(result_file) as f:
        results = json.load(f)

    # weights
    weights = root_pandas.read_root(results.get("weights"))
    # rename columns
    assert weights.columns[1].endswith("EventType")
    weights.rename(
        columns={weights.columns[0]: 'q', weights.columns[1]: 'EventType'},
        inplace=True)

    # train and test sample
    train_samples = results.get("train_samples")
    test_samples = results.get("test_samples")

    return weights, train_samples, test_samples
