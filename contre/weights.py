from numpy import mean


def get_weights(expert_df, normalize_to):
    """Return dataframe with additional weight column.

    The weights are calculated with w = q / (1 - q). Name of the weight column
    is "weight". The weights should be normalized to match the ratio of
    `data / mc` of the samples used for training.

    Parameters:
        expert (pd.DataFrame): dataframe with the classifier output
            for MC (expert should contain only MC classifier output).
        normalize_to (float): normalize the weights,
            if 0: no normalization.
    """
    key_xml = expert_df.keys()[0]
    key_EventType = expert_df.keys()[1]
    assert key_EventType.endswith('EventType')

    expert_df['weight'] = ((expert_df[key_xml]) / (1 - expert_df[key_xml]))

    if normalize_to != 0:
        weight_mean = mean(expert_df['weight'])
        resize = normalize_to / weight_mean
        expert_df['weight'] *= resize
    return expert_df
