import enum
import sys
import functions as f
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# print(sys.argv[1])


def write_gen_results(population, filename, iter):
    if iter == 0:   file = open("results/results_" + filename[:-4] + ".txt","w")
    else:           file = open("results/results_" + filename[:-4] + ".txt","a")

    fitnessess = population.get_fitnesses()
    best_in, _, _ = population.get_fitness_data()
    file.write(f"\nIteration {iter} results: \nBest fitness = {fitnessess[best_in]}\n")

    for i in range(len(population.individuals)):
        file.write(f"\nIndividual {i}, Fitness = {fitnessess[i]}")

def iteration(population, n_next_gen):
    fits = population.get_fitnesses()
    fits = np.argsort(fits)[:n_next_gen]

    res = f.population(population.G,[population.individuals[i] for i in fits])
    return res


def experiment(filename="bier127.tsp", iterations=1000, interval = 100, pop_size=20, n_next_gen = 5, mutation_rate = 0.1, outputs = True):

    print(f"TSP problem used:   {filename}")
    print(f"Num. iterations:    {iterations}")
    print(f"Interval:           {interval}")

    print(f"Population size:    {pop_size}")
    print(f"Mutation rate:      {mutation_rate}")
    if outputs:  print(f"Show outputs:       True")
    else:           print(f"Show outputs:       False")

    start = time.time()

    G = f.get_graph_from_file(filename)

    #random generation
    population = f.generate_random_population(G,pop_size)

    #1 individual is generated with closest neighbor alg
    # population = f.generate_random_population(G,pop_size-1)
    # population.append_individual(f.individual(G, f.get_edge_list(f.closest_neighbor_alg(G))))

    #loading generation
    # population = f.load_gen("results/"+filename[:-4]+"_gen_14999")

    f.plot_figure(G, f.get_edge_list([]), title=filename, name="results/" + filename[:-4])

    bests=[]
    means=[]

    for i in range(iterations):
        write_gen_results(population,filename,i)

        next_gen = iteration(population, n_next_gen)
        
        i_best, best, mean = next_gen.get_fitness_data()
        bests.append(best)
        means.append(mean)
        
        population = f.create_next_gen(next_gen, pop_size, mutation_rate=mutation_rate)

        if i%interval==0:  
            if outputs:
                f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+ filename[:-4] + "_"+ str(i))
            print("Iteration",i)

    write_gen_results(population,filename,i)
    population.save_gen(f"{filename[:-4]}_gen_{i}")
    f.plot_figure(G, next_gen.individuals[i_best].edge_list, name="results/"+ filename[:-4] + "_"+ str(i))

    took = time.time()-start
    # print(f"It took {took:.1f} seconds")

    plt.clf()
    plt.plot(bests, label="Best fitness")
    plt.plot(means, label="Mean")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"Population size: {pop_size}, {took:.1f} seconds")
    plt.savefig("results/" + filename[:-4] + "_progress_" + str(iterations) + "_popsize_" + str(pop_size) +".png")
    plt.close()

    # for i in range(len(next_gen.individuals)):
    #     f.plot_figure(G, next_gen.individuals[i].edge_list, name=str(i))
