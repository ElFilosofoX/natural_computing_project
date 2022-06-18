import pandas as pd
import numpy as np


def read_fitness(file):
    return (
        pd.read_csv(file)["fitness"].to_numpy()[:-2]
    )


sym_ac = read_fitness("csv/aco_symmetric_result.csv")
sym_ec = read_fitness("csv/ec_symmetric_result.csv")
sym_op = read_fitness("csv/op_symmetric_result.csv")

sym_ac = pd.DataFrame({
    "mean": np.mean(sym_ac / sym_op),
    "std": np.std(sym_ac / sym_op),
}, index=["aco"])

sym_ec = pd.DataFrame({
    "mean": np.mean(sym_ec / sym_op),
    "std": np.std(sym_ec / sym_op),
}, index=["ec"])

df = pd.concat([
    sym_ac, sym_ec
])

df.round(2).to_csv("csv/sym_comparison_mean.csv")

asym_ac = read_fitness("csv/aco_asymmetric_result.csv")
asym_ec = read_fitness("csv/ec_asymmetric_result.csv")
asym_op = read_fitness("csv/op_asymmetric_result.csv")

asym_ac = pd.DataFrame({
    "mean": np.mean(asym_ac / asym_op),
    "std": np.std(asym_ac / asym_op),
}, index=["aco"])

asym_ec = pd.DataFrame({
    "mean": np.mean(asym_ec / asym_op),
    "std": np.std(asym_ec / asym_op),
}, index=["ec"])

df = pd.concat([
    asym_ac, asym_ec
])

df.round(2).to_csv("csv/asym_comparison_mean.csv")
