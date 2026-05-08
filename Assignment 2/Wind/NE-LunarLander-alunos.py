import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
TEST_EPISODES = 1000
STEPS = 500

NUM_PROCESSES = int(os.cpu_count()) # type: ignore
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 200
NUMBER_OF_GENERATIONS = 80
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1

ELITE_SIZE = 1

# Tournament size for parent selection.
TOURNAMENT_SIZE = 3

def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y) # returns a value inbetween -1 and 1 for each engine (main, lateral)
    return x

def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation_history):
    """
    Penalize and Reward observations
    After multiple trys we found it to be best to evaluate fitness only at the end (last observation).
    Weights were given to each behaviour accordingly, they were chosenn thro logical reasoning and trial and error.
    Maximum fitness: 1020
    """
    fitness = 0.0
    last_obs = observation_history[-1]
    success = check_successful_landing(last_obs)
    
    x = last_obs[0]
    y = last_obs[1]
    vx = last_obs[2]
    vy = last_obs[3]
    theta = last_obs[4]
    v_theta = last_obs[5]
    left_leg = last_obs[6]
    right_leg = last_obs[7]
    
    # Rewards 
    fitness += (left_leg + right_leg) * 10.0 # leg contact reward
    # Greatly reward a successful landing
    if success: fitness += 1000

    # Penalizations
    fitness -= (x**2 + y**2) * 20.0 # distance penalisation
    fitness -= (vx**2 + vy**2) * 100.0 # velocity penalisation
    fitness -= (theta**2+ v_theta**2) * 50.0 # angle penalisation

    return fitness, success

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    observation_history = [observation]
    for _ in range(STEPS):
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        
        observation_history.append(observation)

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(observation_history)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode =None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    """
    Select the best fitness in between k(Tournament size) random individuals
    This gurantees variability whilst keeping selection pressure
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)

def crossover(p1, p2):
    # 50/50 for each gene 
    child = copy.deepcopy(p1)
    for i in range(GENOTYPE_SIZE):
        if random.random() < 0.5:
            child['genotype'][i] = p2['genotype'][i]
    child['fitness'] = None
    return child

def mutation(p):
    """
    Gaussian (uncorrelated) mutation.

    Each gene is perturbed independently with probability PROB_MUTATION
    by adding noise drawn from N(0, STD_DEV). This operator explores the
    neighbourhood of the current point in genotype space rather than
    making large random jumps.

    PROB_MUTATION = 1/GENOTYPE_SIZE means on average exactly one gene
    is mutated per individual.
    """
    mutant = copy.deepcopy(p)
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            mutant['genotype'][i] += random.gauss(0, STD_DEV)
            mutant['genotype'][i] = max(-1.0, min(1.0, mutant['genotype'][i]))
    mutant['fitness'] = None
    return mutant
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':

    # Personalized Settings   
    evolve = False
    render_mode = None
    both = True
    #render_mode = 'human'
    
    n_runs = 5

    if evolve or both:
        #evolve individuals
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(n_runs):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    if not evolve or both:
        #test evolved individuals
        for j in range (n_runs):
            filename = 'log' + str(j) + '.txt'
            bests = load_bests(filename)
            b = bests[-1]
            SHAPE = b[1]
            ind = b[2]
                
            ind = {'genotype': ind, 'fitness': None}
                
                
            ntests = TEST_EPISODES

            fit, success = 0, 0
            for i in range(1,ntests+1):
                f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
                fit += f
                success += s

            print('Log ' + str(j) + ':')
            print(fit/ntests, (success/ntests)* 100) # for better interpretation 