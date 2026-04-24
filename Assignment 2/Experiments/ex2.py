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

# PROB_MUTATION is the per-gene probability of mutation.
# With GENOTYPE_SIZE genes, 1/GENOTYPE_SIZE means on average 1 gene mutates
# per individual. This is the classic "bit flip" rate adapted for real values.
# A higher value (e.g. 0.05) causes more disruption — useful for escaping
# local optima but can destroy good solutions.
PROB_MUTATION = 1.0 / GENOTYPE_SIZE
STD_DEV = 0.1  # Standard deviation of Gaussian noise applied during mutation.
               # Small enough not to destroy structure, large enough to explore.

# Tournament size for parent selection.
# k=3 balances selection pressure: higher k → more pressure (fitter individuals
# win more often), which speeds convergence but risks premature convergence.
TOURNAMENT_SIZE = 3

ELITE_SIZE = 1  # Number of best individuals carried over unchanged each generation.


def network(shape, observation, ind):
    """Computes the output of the neural network given the observation and genotype."""
    x = observation[:]
    offset = 0
    for i in range(1, len(shape)):
        layer_size = shape[i]
        input_size = len(x)
        y = np.zeros(layer_size)
        for j in range(layer_size):
            for k in range(input_size):
                y[j] += x[k] * ind[offset + k + j * input_size]
        offset += input_size * layer_size
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
    Composite fitness function combining several landing quality signals.

    Components:
      - Horizontal distance to pad (x): penalise lateral drift.
      - Vertical velocity (vy): reward soft descent, penalise hard impact.
      - Tilt angle (theta): penalise leaning.
      - Leg contact bonus: reward actually landing (legs touching ground).
      - Large bonus for a fully successful landing.

    Why this instead of just -abs(x) - abs(y)?
    The original function only looks at position in the second-to-last frame.
    That gives almost no gradient signal for an agent that crashes in the
    middle — all crashes look equally bad. By also penalising velocity and
    angle we give the EA more to work with throughout training, guiding it
    toward stable hover postures even before it learns to land cleanly.
    """
    last_obs = observation_history[-1]

    x       = last_obs[0]   # Horizontal position (0 = pad centre)
    vy      = last_obs[3]   # Vertical velocity (negative = falling)
    theta   = last_obs[4]   # Tilt angle (radians)
    cl      = last_obs[6]   # Left leg contact
    cr      = last_obs[7]   # Right leg contact

    # Distance penalty — strong signal for lateral alignment
    dist_penalty   = -abs(x) * 2.0

    # Velocity penalty — encourage gentle descent (vy < 0 means falling)
    vel_penalty    = -abs(vy) * 0.5

    # Tilt penalty — encourage upright posture
    tilt_penalty   = -abs(theta) * 1.0

    # Leg contact reward — partial credit for at least touching down
    contact_bonus  = (cl + cr) * 0.5

    # Full landing bonus — large reward for satisfying all criteria
    success = check_successful_landing(last_obs)

    fitness = dist_penalty + vel_penalty + tilt_penalty + contact_bonus

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
            turbulence_power=TURBULENCE_POWER
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
    """Evaluates individuals until it receives None. Runs on multiple processes."""
    env = gym.make(
        "LunarLander-v3",
        render_mode=None,
        continuous=True,
        gravity=GRAVITY,
        enable_wind=ENABLE_WIND,
        wind_power=WIND_POWER,
        turbulence_power=TURBULENCE_POWER
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
    """Generates the initial population with random genotypes in [-1, 1]."""
    population = []
    for i in range(POPULATION_SIZE):
        genotype = [random.uniform(-1, 1) for _ in range(GENOTYPE_SIZE)]
        population.append({'genotype': genotype, 'fitness': None})
    return population


def parent_selection(population):
    """
    Tournament selection: sample k individuals at random, return the fittest.

    Why tournament over roulette wheel?
    - Works correctly with negative fitness values (roulette requires positives).
    - Selection pressure is directly tunable via TOURNAMENT_SIZE without
      fitness scaling tricks.
    - Less sensitive to outliers: a single very-high-fitness individual doesn't
      dominate the way it would with fitness-proportional selection.
    """
    tournament = random.sample(population, TOURNAMENT_SIZE)
    winner = max(tournament, key=lambda ind: ind['fitness'])
    return copy.deepcopy(winner)


def crossover(p1, p2):
    """
    Uniform crossover: each gene is independently inherited from p1 or p2
    with equal probability (50/50).

    Why uniform over single-point or two-point?
    - Neural network weights don't have a natural positional meaning —
      gene 47 and gene 48 are not necessarily "related". Single-point
      crossover preserves spatial blocks, which is useful for bit-strings
      encoding a structured solution, but less so for flat weight vectors.
    - Uniform crossover explores more of the recombination space and is
      standard practice for real-valued neuroevolution.
    - Offspring inherits exactly half its genes from each parent on average,
      giving a true blend of both parents' behaviours.
    """
    child_genotype = [
        p1['genotype'][i] if random.random() < 0.5 else p2['genotype'][i]
        for i in range(GENOTYPE_SIZE)
    ]
    return {'genotype': child_genotype, 'fitness': None}


def mutation(p):
    """
    Gaussian mutation: each gene is perturbed by N(0, STD_DEV) with
    probability PROB_MUTATION.

    Why Gaussian perturbation over reset mutation?
    - Reset mutation (replace gene with a random value) destroys all
      information in that gene. Gaussian perturbation keeps the gene
      close to its current value, making small improvements more likely
      than large random jumps — this is critical when the population is
      already reasonably fit.
    - STD_DEV controls the step size. The value 0.1 is small relative
      to the [-1,1] initialisation range, so each mutation step is a
      fine-grained local search move.
    - PROB_MUTATION = 1/GENOTYPE_SIZE means roughly one gene is mutated
      per individual per generation, maintaining diversity without
      destroying good solutions.
    """
    mutant = copy.deepcopy(p)
    for i in range(GENOTYPE_SIZE):
        if random.random() < PROB_MUTATION:
            mutant['genotype'][i] += random.gauss(0, STD_DEV)
            # Optionally clamp to a reasonable range to prevent weight explosion
            # mutant['genotype'][i] = max(-5.0, min(5.0, mutant['genotype'][i]))
    mutant['fitness'] = None
    return mutant


def survival_selection(population, offspring):
    """
    (μ + λ) selection with elitism.

    The elite individuals from the current population are re-evaluated
    (to account for stochastic fitness) and then compete with offspring.
    This ensures the best discovered solution is never lost.
    """
    offspring.sort(key=lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key=lambda x: x['fitness'], reverse=True)
    return new_population


def evolution():
    """Main evolutionary loop."""
    # Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(
            Process(target=evaluate, args=(evaluationQueue, evaluatedQueue))
        )
        evaluation_processes[-1].start()

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

    # Shut down worker processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()

    return bests


def load_bests(fname):
    """Load bests from file."""
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
