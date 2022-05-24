import numpy as np
import networkx as nx
import tsplib95
import matplotlib.pyplot as plt
import math
from random import sample



def generate_random_inividual(G):
    path = generate_random_path(G)
    return individual(G, get_edge_list(path))

def generate_random_population(G,pop_size=10):
    return population(G,[generate_random_inividual(G) for i in range(pop_size)])

def mutate(ind):
    path = ind.get_path()
    path = swap_positions_path(path)
    return individual(ind.G,get_edge_list(path))

def crossover(parent1, parent2):
    #it was supposed to generate 2 childs but for now it only generates 1
    if np.random.rand()>0.5:
        parent1, parent2 = parent2, parent1

    child = []
    cuts = [np.random.randint(len(parent1)), np.random.randint(len(parent1))]
    
    for i in range(min(cuts), max(cuts)):
        child.append(parent1[i])

    child = child + [i for i in parent2 if i not in child]

    return np.array(child)

def create_next_gen(pop, target_size, mutation_rate=True):
    #maybe add random individuals to crossbreed can improve performance
    if len(pop.individuals)==0: return generate_random_population(pop.G)
    gen= pop
    if mutation_rate==True: mutation_rate = 1/target_size

    n = target_size - len(pop.individuals)
    n_inbred = math.floor(n/3)
    n_random = n-n_inbred

    #inbreedings
    for i in range(n_inbred):
        parents = np.random.choice(pop.individuals,2,replace=False)

        offspring = crossover(parents[0].get_path(),parents[1].get_path())
        offspring = individual(pop.G, get_edge_list(offspring))
        while np.random.rand()<mutation_rate:
            offspring = mutate(offspring)
        
        gen.append_individual(offspring)

    #random crossbreedings    
    # for i in range(n_random-1): #add this line to use the closest neighbour down
    for i in range(n_random):         
        parent = np.random.choice(pop.individuals)
        offspring = crossover(parent.get_path(),generate_random_path(pop.G))
        offspring = individual(pop.G, get_edge_list(offspring))
        while np.random.rand()<mutation_rate:
            offspring = mutate(offspring)
        
        gen.append_individual(offspring)
    
    # offspring = crossover(np.random.choice(pop.individuals).get_path(),closest_neighbour_alg(pop.G))
    # offspring = individual(pop.G, get_edge_list(offspring))
    # while np.random.rand()<mutation_rate:
    #     offspring = mutate(offspring)
    
    # gen.append_individual(offspring)

    return gen

def node_positions(G):
    dictionary = nx.get_node_attributes(G,'coord')
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

def closest_neighbour_alg(G):
    N = len(G.nodes)
    arr = np.arange(N) + 1

    visit = [np.random.choice(arr)]

    while len(arr) > 0:
        next_city = 0
        min_fit = np.inf
        for i in arr:
            if i not in visit:
                fit = get_distance(G.nodes[i],G.nodes[visit[-1]])
                if  fit < min_fit:  
                    min_fit = fit
                    next_city=i

        if next_city == 0: break
        visit.append(next_city)
    
    return np.array(visit)

def swap_positions_path(path):
    pos = [0,0]
    while pos[0]==pos[1]:
        pos = np.random.randint(len(path), size=2)

    path[pos[0]], path[pos[1]] = path[pos[1]], path[pos[0]]
    return path

def generate_random_path(G):
    arr=np.array(G.nodes)
    np.random.shuffle(arr)
    return arr

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

    def get_paths(self):
        return [i.get_path() for i in self.individuals]

    def get_fitness_data(self):
        fits = self.get_fitnesses()
        return np.argmin(fits), np.amin(fits),np.mean(fits)

    def append_individual(self, ind):
        self.individuals.append(ind)
