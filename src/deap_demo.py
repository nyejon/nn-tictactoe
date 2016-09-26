
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
rng = np.random


print __file__

#create the types
#the types are created through the *creator* and define the following:
# a) base type for individuals (a simple python list)
# b) the fitness function (a base fitness function provided by deap, with weights of our choosing)
#a minimizing fitness applies negative weights to the fitness evaluation, a maximizing function applies positive weights
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)


#create the individuals
#there seems to be some reflection shit going down where the "Individual" shit created previously is now a python class
INDIVIDUAL_SIZE = 10
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=INDIVIDUAL_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#create the operators
#the individuals have already been put into the toolbox, now to put some of the usual GA operations in there

#the evaluation function measures how close we are to our ideal individual (all elements = 1)
def evaluate(individual): #the evaluation function is the bread and butter of a GA
    #check the trailing comma in the return, everything is a tuple in deap (sometimes)
    return (reduce(lambda x,y: x+y, ((1 - i)*(1 - i) for i in individual)),) #need to accumulate (sum) the errors (squared)
    #return (reduce(lambda x,y: x*y, (np.abs(1 - i) for i in individual)),) #mustn't multiply the errors together
    #return sum(individual), #this fitness function resulted in unbounded growth of the elements 

toolbox.register("mate", tools.cxTwoPoint) #use a built-in function for performing crossover / mating
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) #built-in function for mutation
toolbox.register("select", tools.selTournament, tournsize=3) #built-in function for selecting fittest individuals
toolbox.register("evaluate", evaluate) #set the evaluation function to the one we defined above

#of course, at this point nothing has actually happened because we've just been putting things into the
#mystical "toolbox" and haven't actually run any GA stuff. Time to change that.

def print_pop():
    print "Population:"
    print "\n".join("{:02d}: ".format(n) + ", ".join("{0:0.2f}".format(j) for j in i) for n, i in enumerate(pop))

def print_fitness():
    print "Fitness:"
    #print ", ".join("F{0:02d}: {1:05.2f}".format(n, i[0]) for n, i in enumerate(fitnesses))
    print ", ".join("F{0:02d}: {1:05.2f}".format(n, i.fitness.values[0]) for n, i in enumerate(pop))

pop = toolbox.population(n=50)
CXPB, MUTPB, NGEN = 0.5, 0.2, 40

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print "\Initial population and fitnesses: {}\n"
print_pop()
print_fitness()

for g in range(NGEN):
    print "\nGeneration: {}\n".format(g)
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
    print "\n".join("{:02d}: ".format(n) + ", ".join("{0:0.2f}".format(j) for j in i) for n, i in enumerate(pop[:1]))

print "*" * 80
print "We're done!"
print_pop()

fitnesses = map(toolbox.evaluate, pop)
print_fitness()
fittest = min(enumerate(fitnesses), key=lambda x: x[1][0])
print repr(fittest)
print "The fittest individual was #{0} with a fitness of {1:0.2f}.".format(fittest[0], fittest[1][0])
print "The individual's gene was:"
print ", ".join("{0:0.2f}".format(j) for j in pop[fittest[0]])

#if __name__ == "__main__":
#    main()




