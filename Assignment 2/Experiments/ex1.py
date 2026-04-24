import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
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
SHAPE = (nInputs, 12, nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1] * SHAPE[i]

POPULATION_SIZE = 100
NUMBER_OF_GENERATIONS = 100
PROB_CROSSOVER = 0.9

PROB_MUTATION = 1.0 / GENOTYPE_SIZE
STD_DEV = 0.1

ELITE_SIZE = 1

# Tournament size for parent selection.
# k=3 gives good pressure without too much greediness.
TOURNAMENT_SIZE = 3


def network(shape, observation, ind):
    """Computes the output of the neural network given the observation and genotype."""
    x = observation[:]
    for i in range(1, len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k] * ind[k + j * len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    """Checks the success of the landing based on the observation."""
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
    Shaped fitness function that rewards the lander throughout the whole episode,
    not just at the final step.

    Components (all contribute negatively when bad):
      - Horizontal distance from pad: penalise being far from x=0.
      - Vertical distance from ground: rewards descending towards y=0.
      - Horizontal velocity: rewards slow lateral movement.
      - Vertical velocity: rewards gentle descent (not crashing).
      - Angle: rewards staying upright.
      - Angular velocity: rewards rotational stability.
      - Leg contact bonus: extra reward each step a leg touches down.

    Using a mean-over-trajectory approach gives a dense training signal
    compared to only checking the final observation, which is sparse
    and forces the EA to stumble upon good behaviours by chance.
    """
    fitness = 0.0
    for obs in observation_history:
        x       = obs[0]   # horizontal position   (0 = pad)
        y       = obs[1]   # vertical position      (0 = ground)
        vx      = obs[2]   # horizontal velocity
        vy      = obs[3]   # vertical velocity
        theta   = obs[4]   # angle
        omega   = obs[5]   # angular velocity
        cl      = obs[6]   # left  leg contact
        cr      = obs[7]   # right leg contact

        # Penalise distance from pad (horizontal and vertical)
        fitness -= abs(x)
        fitness -= abs(y)

        # Penalise velocities — smooth, slow approach is better
        fitness -= 0.5 * abs(vx)
        fitness -= 0.5 * abs(vy)

        # Penalise tilt and spin
        fitness -= 0.3 * abs(theta)
        fitness -= 0.2 * abs(omega)

        # Reward each leg touching the ground
        fitness += 0.5 * (cl + cr)

    # Normalise by episode length so fitness is comparable
    # across episodes that terminated at different steps
    fitness /= len(observation_history)

    success = check_successful_landing(observation_history[-1])
    return fitness, success


def simulate(genotype, render_mode=None, seed=None, env=None):
    """Simulates an episode of Lunar Lander, evaluating an individual."""
    env_was_none = env is None
    if env is None:
        env = gym.make(
            "LunarLander-v3",
            render_mode=render_mode,
            continuous=True,
            gravity=GRAVITY,
            enable_wind=ENABLE_WIND,
            wind_power=WIND_POWER,
            turbulence_power=TURBULENCE_POWER,
        )

    observation, info = env.reset(seed=seed)
    observation_history = [observation]

    for _ in range(STEPS):
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)
        observation_history.append(observation)
        if terminated or truncated:
            break

    if env_was_none:
        env.close()

    return objective_function(observation_history)


def evaluate(evaluationQueue, evaluatedQueue):
    """Evaluates individuals until it receives None. Runs on a worker process."""
    env = gym.make(
        "LunarLander-v3",
        render_mode=None,
        continuous=True,
        gravity=GRAVITY,
        enable_wind=ENABLE_WIND,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER,
    )
    while True:
        ind = evaluationQueue.get()
        if ind is None:
            break
        ind['fitness'] = simulate(ind['genotype'], seed=None, env=env)[0]
        evaluatedQueue.put(ind)
    env.close()


def evaluate_population(population):
    """Evaluates a list of individuals using multiple processes."""
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop


def generate_initial_population():
    """Generates the initial population with weights in [-1, 1]."""
    population = []
    for i in range(POPULATION_SIZE):
        genotype = [random.uniform(-1, 1) for _ in range(GENOTYPE_SIZE)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def parent_selection(population):
    """
    Tournament selection with k=3.

    Randomly sample k individuals from the population and return a deep copy
    of the one with the highest fitness. This applies selection pressure
    (better individuals win more often) while keeping some diversity
    (the best individual does not always win). Compared to roulette-wheel
    selection it is simple, parameter-light, and robust to negative fitnesses.
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)


def crossover(p1, p2):
    """
    Uniform crossover.

    For each gene position, independently choose with probability 0.5
    whether the offspring inherits the gene from p1 or p2.
    This thoroughly mixes information from both parents, which is
    especially useful for neural network weights where any gene can
    interact with any other — single-point crossover would leave large
    correlated blocks intact and explore more slowly.
    """
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
    by adding noise drawn from N(0, STD_DEV). This is the standard
    mutation operator for real-valued EAs (ES-style): it explores the
    neighbourhood of the current point in genotype space rather than
    making large random jumps, which is important for fine-tuning
    neural network weights late in the search.

    PROB_MUTATION = 1/GENOTYPE_SIZE means on average exactly one gene
    is mutated per individual — a common rule of thumb that balances
    exploration and exploitation.
    """
    mutant = copy.deepcopy(p)
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            mutant['genotype'][i] += random.gauss(0, STD_DEV)
    mutant['fitness'] = None
    return mutant


def survival_selection(population, offspring):
    """
    Elitism: keep the top ELITE_SIZE individuals from the current population
    (re-evaluated to account for stochasticity), then fill the rest of the
    new population from the offspring ranked by fitness.
    """
    offspring.sort(key=lambda x: x['fitness'], reverse=True)
    # Re-evaluate the elite to reduce noise in their fitness estimate
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key=lambda x: x['fitness'], reverse=True)
    return new_population


def evolution():
    """Main evolutionary loop."""
    # Create evaluation worker processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(
            Process(target=evaluate, args=(evaluationQueue, evaluatedQueue))
        )
        evaluation_processes[-1].start()

    # Initialise and evaluate the population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key=lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)

    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []

        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)
            else:
                ni = parent_selection(population)

            ni = mutation(ni)
            offspring.append(ni)

        offspring = evaluate_population(offspring)
        population = survival_selection(population, offspring)

        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]:.4f}')

    # Signal workers to stop
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()

    return bests


def load_bests(fname):
    """Load bests from log file."""
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append((eval(fitness), eval(shape), eval(genotype)))
    return bests


if __name__ == '__main__':

    # Pick a setting from below:

    # -- to evolve the controller --
    evolve = False
    render_mode = None

    # -- to test the evolved controller without visualisation --
    # evolve = False
    # render_mode = None

    # -- to test the evolved controller with visualisation --
    # evolve = False
    # render_mode = 'human'

    if evolve:
        n_runs = 5
        seeds = [
            964, 952, 364, 913, 140, 726, 112, 631, 881, 844,
            965, 672, 335, 611, 457, 591, 551, 538, 673, 437,
            513, 893, 709, 489, 788, 709, 751, 467, 596, 976,
        ]
        for i in range(n_runs):
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

    else:
        filename = 'log0.txt'
        bests = load_bests(filename)
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]

        ind = {'genotype': ind, 'fitness': None}

        ntests = TEST_EPISODES
        fit, success = 0, 0
        for i in range(1, ntests + 1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed=None)
            fit += f
            success += s
        print(f'Mean fitness: {fit/ntests:.4f}  |  Success rate: {success/ntests:.2%}')
