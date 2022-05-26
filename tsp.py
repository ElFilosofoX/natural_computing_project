import enum
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
    iterations=10000
    print(f"Num. iterations:    {iterations}")
    interval = iterations/10
    print(f"Interval:           {interval}")
    pop_size=50
    if pop_size%2 == 1: 
        print("Population size must be even")
        sys.exit()
    n_next_gen = math.floor(pop_size/3)
    print(f"Population size:    {pop_size}")
    mutation_rate = 0.1
    print(f"Mutation rate:      {mutation_rate}")
    early_stop = False
    if early_stop:  print(f"Early stop:         True")
    else:           print(f"Early stop:         False")
    outputs = True
    if outputs:  print(f"Show outputs:       True")
    else:           print(f"Show outputs:       False")
    

elif len(sys.argv)!=7:
    print("Command run as follows: python tsp.py filename iterations pop_size mutation_rate early_stop")
    sys.exit()


def write_gen_results(population, iter):
    if iter == 0:   file = open("results/results" + filename[:-4] + ".txt","w")
    else:           file = open("results/results" + filename[:-4] + ".txt","a")

    fitnessess = population.get_fitnesses()
    best_in, _, _ = population.get_fitness_data()
    file.write(f"\nIteration {iter} results: \nBest fitness = {fitnessess[best_in]}\n")

    for i in range(len(population.individuals)):
        file.write(f"\nIndividual {i}, Fitness = {fitnessess[i]}")

def iteration(population):
    fits = population.get_fitnesses()
    fits = np.argsort(fits)[:n_next_gen]

    res = f.population(G,[population.individuals[i] for i in fits])
    return res

start = time.time()

G = f.get_graph_from_file(filename)

#random generation
population = f.generate_random_population(G,pop_size)

#1 individual is generated with closest neighbour alg
# population = f.generate_random_population(G,pop_size-1)
# population.append_individual(f.individual(G, f.get_edge_list(f.closest_neighbour_alg(G))))

#loading generation
# population = f.load_gen("results/"+filename[:-4]+"_gen_14999")

f.plot_figure(G, f.get_edge_list([]), title=filename, name="results/" + filename[:-4])

bests=[]
means=[]

for i in range(iterations):
    write_gen_results(population,i)

    next_gen = iteration(population)
    next_gen_paths = next_gen.get_paths()
    
    i_best, best, mean = next_gen.get_fitness_data()
    bests.append(best)
    means.append(mean)
    
    population = f.create_next_gen(next_gen, pop_size, mutation_rate=mutation_rate)

    if i%interval==0:  
        if early_stop and i>0 and best >= bests[math.floor(i-(interval/2))]: 
            print("Progress has stagnated")
            break
        if outputs:
            f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+ filename[:-4] + "_"+ str(i))
        print(i)

write_gen_results(population,i)
population.save_gen(f"{filename[:-4]}_gen_{i}")
f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+ filename[:-4] + "_"+ str(i))

took = time.time()-start
print(f"It took {took:.1f} seconds")

plt.clf()
plt.plot(bests, label="Best fitness")
plt.plot(means, label="Mean")
plt.legend()
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title(f"Population size: {pop_size}, {took:.1f} seconds")
plt.savefig("results/progress.png")
plt.close()

# for i in range(len(next_gen.individuals)):
#     f.plot_figure(G, next_gen.individuals[i].edge_list, name=str(i))
