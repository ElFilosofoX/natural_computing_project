import sys
import functions as f
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# print(sys.argv[1])
class experiment_tsp:

    def __init__(self, filename, iterations=1000, interval=100, pop_size=100, mutation_rate=0.1, early_stop=False, outputs=True):
        self.filename = filename
        self.iterations = iterations
        self.pop_size = pop_size
        self.interval = interval
        self.mutation_rate = mutation_rate
        self.early_stop = early_stop
        self.outputs = outputs
    

def write_gen_results(population, iter):
    if iter == 0:   file = open("results/results.txt","w")
    else:           file = open("results/results.txt","a")

    fitnessess = population.get_fitnesses()
    best_in, _ = population.get_best_fitness()
    file.write(f"\nIteration {iter} results: \nBest fitness = {fitnessess[best_in]}\n")

    for i in range(len(population.individuals)):
        file.write(f"\nIndividual {i}, Fitness = {fitnessess[i]}")

def iteration(population):
    fits = population.get_fitnesses()

    p1 = np.argmin(fits)
    fits = np.delete(fits, p1)
    p2 = np.argmin(fits)

    return f.population(G, 
    [
        f.individual(G, population.individuals[p1].edge_list),
        f.individual(G, population.individuals[p2].edge_list)
    ])

start = time.time()

G = f.get_graph_from_file(filename)
population = f.generate_random_population(G,pop_size)

f.plot_figure(G, f.get_edge_list([]), title=filename, name="results/base")

hist=[]
for i in range(iterations):
    write_gen_results(population,i)

    next_gen = iteration(population)
    next_gen_paths = next_gen.get_paths()
    
    i_best, best = next_gen.get_best_fitness()
    hist.append(best)
    
    population = f.create_next_gen(next_gen, pop_size, mutation_rate=mutation_rate)

    if i%interval==0:  
        if early_stop and i>0 and best >= hist[math.floor(i-(interval/2))]: 
            print("Progress has stagnated")
            break
        if outputs:
            f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+str(i))
        print(i)

write_gen_results(population,i)
f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+str(i))

print(f"It took {time.time()-start:.1f} seconds")

plt.clf()
plt.plot(hist)
plt.savefig("results/progress.png")
plt.close()