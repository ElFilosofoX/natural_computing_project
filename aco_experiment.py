import aco
import tsplib95

import pandas as pd
import networkx as nx
import numpy as np

from pathlib import Path

FITNESS = aco.get_fitness
ANTS = 50
RNG = np.random.default_rng(123)
MAX_TIME = 60


def run_all(data_dir):
    probs = Path(data_dir).glob("*.tsp")
    return pd.DataFrame(
        {
            "fitness": run_aco(p, FITNESS, ANTS, RNG, MAX_TIME),
            "dataset": p
        }
        for p in probs
    )


def run_aco(filename, fitness, ants, rng, max_time):
    prob = tsplib95.load(filename)
    g = nx.convert_node_labels_to_integers(prob.get_graph())
    times, fitnesses, paths = aco.aco(g, fitness, ants, rng, max_time)
    return np.min(fitnesses)
