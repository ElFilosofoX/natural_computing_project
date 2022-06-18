import networkx as nx
import numpy as np
import time
import tsplib95

import pandas as pd

from pathlib import Path


def aco(g, fitness, ants, rng, max_time):
    """
    Ant colony optimization algorithm.

    Parameters
    ----------
    g: graph
        networkx graph representing the TSP problem
    fitness: function
        Function that calculates fitness of a path.
    ants: int
        Number of ants.
    rng: rng_generator
        rng generator
    max_time: float
        Time to run the algorithm.

    Returns
    -------
    times: list of floats
        A list containing the runtime for each iteration.
    bests_fit: list of floats
        A list containing the fitness of best path for each iteration.
    bests_path: list of np.array
        A list containing the best path for each iteration.
    """
    start_time = time.time()
    current_time = time.time()
    times = []
    bests_fit = []
    bests_path = []

    nodes = len(g.nodes)
    # pheromone, represented in an adjacency matrix
    parts = np.ones((nodes, nodes), dtype=float) / nodes
    distances = nx.to_numpy_array(g)

    while (current_time - start_time) < max_time:
        paths = np.empty((ants, nodes), dtype=int)
        for ant in range(ants):
            paths[ant, 0] = 0
            visited = np.zeros(nodes, dtype=bool)
            visited[0] = True

            edges = np.ma.MaskedArray(parts[0], visited)
            edges = edges / distances[0]
            for i in range(1, nodes):
                total_pheromone = np.sum(edges)
                acc_pheromone = np.cumsum(edges)

                # we choose which edge to pick
                r = rng.random() * total_pheromone
                next_node = np.digitize(r, acc_pheromone)

                # update path and visited
                paths[ant, i] = next_node
                visited[next_node] = True
                edges = np.ma.MaskedArray(parts[next_node], visited)
                edges = edges / (distances[next_node] + 0.00001)

        # we change from arrays of nodes to arrays of edges, since this is
        # necessary for calculating the fitness
        pairs = np.concatenate(
            [
                paths[..., np.newaxis],
                np.roll(paths, shift=-1, axis=1)[..., np.newaxis]
            ],
            axis=2
        )
        fitnesses = [fitness(g, p) for p in pairs]
        new_parts = np.zeros((nodes, nodes))
        for f in range(len(fitnesses)):
            for i, j in pairs[f]:
                new_parts[i, j] += 1/fitnesses[f]

        # TODO(daan): test different global updates
        parts = parts / np.sum(parts, axis=1)
        new_parts = new_parts / np.sum(new_parts, axis=1)
        equal_parts = np.ones((nodes, nodes)) / nodes
        parts = 0.6 * parts + 0.39 * new_parts + 0.01 * equal_parts

        best_id = np.argmin(fitnesses)
        best_fit = fitnesses[best_id]
        best_path = paths[best_id]

        # if len(bests_fit) == 0 or best_fit < bests_fit[-1]:
        bests_fit.append(best_fit)
        bests_path.append(best_path)
        # else:
        #     bests_fit.append(bests_fit[-1])
        #     bests_path.append(bests_path[-1])

        current_time = time.time()
        t = current_time - start_time
        times.append(t)

    return times, bests_fit, bests_path


def get_fitness(G, edge_list):
    fitness = 0
    for pair in edge_list:
        fitness += G.edges[pair[0], pair[1]]["weight"]
    return fitness


FITNESS = get_fitness
ANTS = 50
RNG = np.random.default_rng(123)
MAX_TIME = 60


def run_from_csv(csv, data_dir, extension):
    return (
        pd.read_csv(csv)
        .assign(
            fitness=lambda frame:
                frame.apply(
                    lambda row:
                    run_aco(data_dir / (str(row["dataset"]) + extension),
                            FITNESS, ANTS, RNG,
                            row["runtime"]),
                    axis=1
                )

        )
    )


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
    times, fitnesses, paths = aco(g, fitness, ants, rng, max_time)
    f = np.min(fitnesses)
    print(f"{filename} done, f = {f}")
    return f
