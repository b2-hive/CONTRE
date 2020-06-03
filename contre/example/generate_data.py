import numpy as np
import pandas as pd
from root_pandas import to_root

if __name__ == "__main__":
    size_mc = 500000
    size_data = 10000
    size_mc_offres = 150000
    size_data_offres = 8000
    frac_a = 0.8
    frac_b = 1 - frac_a

    # GENERATE DATA
    print(
        "Generating the following dataframes:\n"
        "data, componentA, componentB, data_offres and componentA_offres ...")
    # on res
    data = pd.DataFrame()
    componentA = pd.DataFrame()
    componentB = pd.DataFrame()

    # variable1
    tmp_data = np.random.triangular(0, 1, 1, size=int(size_data*frac_a*0.3))
    tmp_data = np.append(
        tmp_data, np.random.normal(0.3, 0.1, int(size_data*frac_b)))
    tmp_data = np.append(
        tmp_data, np.random.uniform(size=int(size_data*frac_a*0.7)))
    data["variable1"] = tmp_data
    data = data.loc[data["variable1"] >= 0]

    componentA["variable1"] = np.random.uniform(size=int(size_mc * frac_a))
    componentB["variable1"] = np.random.normal(
        0.3, 0.1, size=int(size_mc * frac_b))

    # variable2
    data["variable2"] = np.random.uniform(size=len(data))
    componentA["variable2"] = np.random.uniform(size=int(size_mc*frac_a))
    componentB["variable2"] = np.random.uniform(size=int(size_mc*frac_b))

    # candidate and EventType
    data["__candidate__"] = [0]*len(data)
    componentA["__candidate__"] = [0]*len(componentA)
    componentB["__candidate__"] = [0]*len(componentB)

    data["EventType"] = [float(1)]*len(data)
    componentA["EventType"] = [float(0)]*len(componentA)
    componentB["EventType"] = [float(0)]*len(componentB)

    # off res
    data_offres = pd.DataFrame()
    componentA_offres = pd.DataFrame()

    # variable1
    tmp_data = np.random.triangular(
        0, 1, 1, size=int(size_data_offres*frac_a*0.3))
    tmp_data = np.append(
        tmp_data, np.random.uniform(size=int(size_data_offres*frac_a*0.7)))
    data_offres["variable1"] = tmp_data
    componentA_offres["variable1"] = np.random.uniform(
        size=int(size_mc_offres*frac_a))

    # variable2
    data_offres["variable2"] = np.random.uniform(size=len(data_offres))
    componentA_offres["variable2"] = np.random.uniform(
        size=int(size_mc_offres*frac_a))

    # candidate and EventType
    data_offres["__candidate__"] = [0]*len(data_offres)
    componentA_offres["__candidate__"] = [0]*len(componentA_offres)

    data_offres["EventType"] = [float(1)]*len(data_offres)
    componentA_offres["EventType"] = [float(0)]*len(componentA_offres)

    # SAVE DATA
    print("Saving data to 'example_input/<file>.root' ...")

    to_root(data, "example_input/data.root", key="variables")
    to_root(componentA, "example_input/componentA.root", key="variables")
    to_root(componentB, "example_input/componentB.root", key="variables")
    to_root(data_offres, "example_input/data_offres.root", key="variables")
    to_root(componentA_offres, "example_input/componentA_offres.root", key="variables")
