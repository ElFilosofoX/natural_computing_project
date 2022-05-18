import sys
import functions as f
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# print(sys.argv[1])

if len(sys.argv)==1:
    filename="a280.tsp"
    print(f"TSP problem used:   {filename}")
    iterations=50000
    print(f"Num. iterations:    {iterations}")
    interval = 1000
    print(f"Interval:           {interval}")
    pop_size=10
    n_next_gen = math.floor(pop_size/3)
    print(f"Population size:    {pop_size}")
    mutation_rate = 1/pop_size
    print(f"Mutation rate:      {mutation_rate}")
    

elif len(sys.argv)!=5:
    print("Command run as follows: python tsp.py filename iterations pop_size mutation_rate")
    sys.exit()


def write_gen_results(population, iter):
    if iter == 0:   file = open("results/results.txt","w")
    else:           file = open("results/results.txt","a")

    fitnessess = population.get_fitnesses()
    best_in, _ = population.get_best_fitness()
    file.write(f"\nIteration {iter} results: \nBest fitness = {fitnessess[best_in]}\n")

    for i in range(len(population.individuals)):
        file.write(f"\nIndividual {i}, Fitness = {fitnessess[i]}")

def iteration(population):
    min_in, _ = population.get_best_fitness()
    res = f.population(G,[population.individuals[min_in]])
    return res  

start = time.time()

G = f.get_graph_from_file(filename)
population = f.generate_random_population(G,pop_size)

f.plot_figure(G, f.get_edge_list([]), title=filename, name="results/base")

hist=[]
for i in range(iterations):
    write_gen_results(population,i)

    next_gen = iteration(population)
    best = population.individuals[0]
    hist.append(best.get_fitness())
    
    population = f.create_next_gen(next_gen, pop_size)

    if i%interval==0:  
        if i>0 and best.get_fitness() >= hist[math.floor(i-(interval/2))]: 
            print("Progress has stagnated")
            break
        f.plot_figure(G, best.edge_list, name="results/"+str(i))
        print(i)

write_gen_results(population,i)
f.plot_figure(G, best.edge_list, name="results/"+str(i))

print(f"It took {time.time()-start:.1f}")

plt.clf()
plt.plot(hist)
plt.savefig("results/progress.png")

# for i in range(len(next_gen.individuals)):
#     f.plot_figure(G, next_gen.individuals[i].edge_list, name=str(i))
