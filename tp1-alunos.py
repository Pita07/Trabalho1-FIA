import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
#RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

CURRENT_POS = 0
TIME_STOPPED = 0

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
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
        print("Aterragem bem sucedida!")
        return True

    print("Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ) # type: ignore

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
def is_inclined_to_left(observation):
    x = observation[0]
    vx = observation[2]
    theta = observation[4]

    target_theta = (0.75 * x + 1.05 * vx)
    target_theta = np.clip(target_theta, -0.4, 0.4)

    # 0.05 error margin
    return target_theta + 0.05 < theta

def is_inclined_to_right(observation):
    x = observation[0]
    vx = observation[2]
    theta = observation[4]

    target_theta = (0.75 * x + 1.05 * vx)
    target_theta = np.clip(target_theta, -0.4, 0.4)

    # 0.05 error margin
    return target_theta - 0.05 > theta

def is_falling_too_fast(observation):
    x = observation[0]
    vy = observation[3]

    target_vy = abs(x) -0.4
    target_vy = np.clip(target_vy, -0.4, 0)

    return vy < target_vy

def is_rotating_too_fast_to_left(observation):
    omega = observation[5]
    return omega > 0.3

def is_rotating_too_fast_to_right(observation):
    omega = observation[5]
    return omega < -0.3

def is_not_moving_on_ground(observation):
    global CURRENT_POS, TIME_STOPPED
    y = observation[1]

    if y == CURRENT_POS:
        TIME_STOPPED += 1

    CURRENT_POS = y

    return TIME_STOPPED > 10

#Actions
def fire_left_engine():
    return 1

def fire_right_engine():
    return -1

def fire_main_engine_slow():
    return 0.6

def fire_main_engine_full():
    global TIME_STOPPED
    TIME_STOPPED = 0 # reset the counter
    return 1

def dont_fire_engine():
    return 0


def reactive_agent(observation):
    # inclination control
    if is_inclined_to_left(observation):
        side = fire_left_engine()
    elif is_inclined_to_right(observation):
        side = fire_right_engine()
    else:
        side = dont_fire_engine()

    # height control
    if is_falling_too_fast(observation):
        main = fire_main_engine_slow()
    else:
        main = dont_fire_engine()

    # angular velocity control (takes priority over inclination control)
    if is_rotating_too_fast_to_left(observation):
        side = fire_left_engine()
    elif is_rotating_too_fast_to_right(observation):
        side = fire_right_engine()

    # check if we are not moving
    if is_not_moving_on_ground(observation):
        main = fire_main_engine_full()

    return np.array([main, side])
    
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    #print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    

success = 0.0
steps = 0.0
for i in range(EPISODES):
    #st, su = simulate(steps=1000000, policy=keyboard_agent)
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)