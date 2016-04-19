
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from pickle import POP
rng = np.random


print __file__

goal = [1.0 for i in xrange(10)]

def fitness(x):
    return reduce(lambda x,y: x+y, x)

#create the types
#the types are created through the *creator* and define the following:
# a) base type for individuals (a simple python list)
# b) the fitness function (a maximization function provided by deap)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


#create the individuals
#there seems to be some reflection shit going down where the "Individual" shit created previously is now a python class
INDIVIDUAL_SIZE = 10
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#create the operators
#the individuals have already been put into the toolbox, now to put some of the usual GA operations in there

def evaluate(individual): #the evaluation function is the bread and butter of a GA
    return sum(individual), #check the trailing comma, everything is a tuple in deap (sometimes)

toolbox.register("mate", tools.cxTwoPoint) #use a built-in function for performing crossover / mating
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) #built-in function for mutation
toolbox.register("select", tools.selTournament, tournsize=3) #built-in function for selecting fittest individuals
toolbox.register("evaluate", evaluate) #set the evaluation function to the one we defined above

#of course, at this point nothing has actually happened because we've just been putting things into the
#mystical "toolbox" and haven't actually run any GA stuff. Time to change that.

def print_pop():
    print "\n".join("{:02d}: ".format(n) + ", ".join("{0:0.2f}".format(j) for j in i) for n, i in enumerate(pop))

def print_fitness():
    print ", ".join("F{0:02d}: {1:05.2f}".format(n, i[0]) for n, i in enumerate(fitnesses))

pop = toolbox.population(n=50)
CXPB, MUTPB, NGEN = 0.5, 0.2, 40

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
print_pop()
print_fitness()
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = map(toolbox.clone, offspring)

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    print_fitness()

print "*" * 80
print "We're done!"
print_pop()


#if __name__ == "__main__":
#    main()




