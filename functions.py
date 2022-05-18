import numpy as np
import networkx as nx
import tsplib95
import matplotlib.pyplot as plt
import math


def generate_random_inividual(G):
    path = generate__random_path(G)
    return individual(G, get_edge_list(path))

def generate_random_population(G,pop_size=10):
    return population(G,[generate_random_inividual(G) for i in range(pop_size)])

def mutate(ind):
    path = ind.get_path()
    path = swap_positions_path(path)
    return individual(ind.G,get_edge_list(path))

def node_positions(G):
    dictionary = dict()

    for i in range(1,len(G.nodes)+1):
        dictionary[i] = G.nodes[i]['coord']

    return dictionary

def get_graph_from_file(filename):
    problem = tsplib95.load(filename)
    return problem.get_graph()

def get_edge_list(array):
    if array == []: return array
    res=[]

    for i in range(len(array)-1):
        res.append((array[i],array[i+1]))

    res.append((array[-1],array[0]))

    return res

def get_distance(node1, node2):
    return math.dist(node1['coord'],node2['coord'])

def get_fitness(G, edge_list):
    fitness=0
    for pair in edge_list:
        fitness += get_distance(G.nodes[pair[0]],G.nodes[pair[1]])
    return fitness

def get_path_from_edgelist(edge_list):
    return np.array([i[0] for i in edge_list])

def swap_positions_path(path):
    pos = [0,0]
    while pos[0]==pos[1]:
        pos = np.random.randint(len(path), size=2)

    path[pos[0]], path[pos[1]] = path[pos[1]], path[pos[0]]
    return path

def generate__random_path(G):
    arr=np.array(G.nodes)
    np.random.shuffle(arr)
    return arr

def create_next_gen(population, target_size):
    if len(population.individuals)==0: return generate_random_population(population.G)
    gen = population
    best = population.individuals[0]
    n = target_size - len(population.individuals)
    for i in range(n):
        gen.append_individual(mutate(best))

    return gen

def plot_figure(G, edge_list, name="output.png", node_size=20, fig_size=5, title="fitness"):
    plt.clf()
    layout = node_positions(G)
    _, axs = plt.subplots(1, figsize=(fig_size,fig_size))
    # nx.draw_networkx(G, pos=layout, node_size=node_size, with_labels=False, edgelist=edge_list, ax=axs)
    nx.draw_networkx_edges(G, pos=layout, width=0.5,edgelist=edge_list, ax=axs)
    nx.draw_networkx_nodes(G, pos=layout, node_size=node_size,ax=axs)
    plt.axis("off")
    if title == "fitness":  plt.title(f"Fitness: {get_fitness(G,edge_list):0.2f}")
    else:                   plt.title(title)
    plt.savefig(name, dpi=100)
    plt.close()
    

class individual:
    def __init__(self, g, edge_list):
        self.G = g
        self.edge_list = edge_list
    
    def get_fitness(self):
        return get_fitness(self.G, self.edge_list) 

    def set_edge_list(self, new_edge_list):
        self.edge_list = new_edge_list

    def get_path(self):
        return get_path_from_edgelist(self.edge_list)

    def set_path(self, path):
        self.edge_list = get_edge_list(path)


class population:
    def __init__(self, g, individuals):
        self.G = g
        self.individuals = individuals
    
    def get_mean_fitness(self):
        return np.mean([i.get_fitness() for i in self.individuals])

    def get_fitnesses(self):
        return [i.get_fitness() for i in self.individuals]

    def get_best_fitness(self):
        return np.argmin(self.get_fitnesses()), np.amin(self.get_fitnesses())

    def append_individual(self, ind):
        self.individuals.append(ind)
