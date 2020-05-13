import numpy as np
import pandas as pd
from root_pandas import to_root

size_mc = 5000000
size_data = 100000
frac_a = 0.8
frac_b = 1 - frac_a
frac_offres = 0.5

# GENERATE DATA
print(
    "Generating the following dataframes:\n"
    "data, componentA, componentB, data_offres and componentA_offres ...")
# on res
data = pd.DataFrame()
componentA = pd.DataFrame()
componentB = pd.DataFrame()

# variable1
tmp_data = np.random.triangular(-1, 1, 1, size=int(size_data*frac_a*1.3))
tmp_data = np.append(
    tmp_data, np.random.normal(0.3, 0.1, int(size_data*frac_b)))
data["variable1"] = tmp_data
data = data.loc[data["variable1"] >= 0]

componentA["variable1"] = np.random.uniform(size=int(size_mc * frac_a))
componentB["variable1"] = np.random.normal(
    0.3, 0.1, size=int(size_mc * frac_b))

# variable2
data["variable2"] = np.random.uniform(size=len(data))
componentA["variable2"] = np.random.uniform(size=int(size_mc*frac_a))
componentB["variable2"] = np.random.uniform(size=int(size_mc*frac_b))


# off res
data_offres = pd.DataFrame()
componentA_offres = pd.DataFrame()

# variable1
data_offres["variable1"] = np.random.triangular(
    -1, 1, 1, size=int(size_data*frac_a*frac_offres*1.3))
data_offres = data_offres.loc[data_offres["variable1"] > 0]
componentA_offres["variable1"] = np.random.uniform(
    size=int(size_mc*frac_a*frac_offres))

# variable2
data_offres["variable2"] = np.random.uniform(size=len(data_offres))
componentA_offres["variable2"] = np.random.uniform(
    size=int(size_mc*frac_a*frac_offres))

# SAVE DATA
print("Saving data to 'example_input/<file>.root' ...")

to_root(data, "example_input/data.root")
to_root(componentA, "example_input/componentA.root")
to_root(componentB, "example_input/componentB.root")
to_root(data_offres, "example_input/data_offres.root")
to_root(componentA_offres, "example_input/componentA_offres.root")