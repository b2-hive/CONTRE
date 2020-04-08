import root_pandas
from numpy import mean


def get_weights(expert, normalize_to):
    """Return dataframe with additional weight column.

    The weights are calculated with w = q / (1 - q). Name of the weight column
    is "weight". The weights should be normalized to match the ratio of data / mc
    of the samples used for training.

    Parameters:
        expert (pd.DataFrame): dataframe with the classifier output
            for MC (expert should contain only MC classifier output).
        normalize_to (float): normalize the weights,
            if 0: no normalization.
    """
    key_xml = expert.keys()[0]
    key_EventType = expert.keys()[1]
    assert key_EventType.endswith('EventType')
    # assert expert.at[0, key_EventType] == 0  # removed output belongig to data?

    expert['weight'] = ((expert[key_xml]) / (1 - expert[key_xml]))

    if normalize_to != 0:
        weight_mean = mean(expert['weight'])
        resize = normalize_to / weight_mean
        expert['weight'] *= resize
    return expert
