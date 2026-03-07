import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
#RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

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
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
##TODO: Defina as suas perceções aqui
def is_left_of_pad(observation):
    x = observation[0]
    return x < -0.2

def is_right_of_pad(observation):
    x = observation[0]
    return x > 0.2

def has_stable_velocity(observation):
    vy = observation[3]
    return vy > -0.4

def has_stable_orientation(observation):
    theta = observation[4]
    return abs(theta) < np.deg2rad(18)

def is_near_starting_point(observation):
    x = observation[0]
    y = observation[1]
    return (-0.2 < x < 0.2) and (1.45 > y > 1.1)

def is_near_landing_pad(observation):
    x = observation[0]
    y = observation[1]
    return (-0.2 < x < 0.2) and (y < 0.3)

def tilted_to_left(observation):
    theta = observation[4]
    return 0.01 < theta < 0.1

def tilted_to_right(observation):
    theta = observation[4]
    return -0.1 < theta < -0.01

def moving_left(observation):
    vx = observation[2]
    return vx < -0.1

def moving_right(observation):
    vx = observation[2]
    return vx > 0.1

def has_upwards_momentum(observation):
    vy = observation[3]
    return vy > -0.05

#Actions
##TODO: Defina as suas ações aqui
def go_left():
    return np.array([0, -1])

def go_right():
    return np.array([0, 1])

def go_up():
    return np.array([1, 0])

def stabilize_right():
    return np.array([1, 1])

def stabilize_left():
    return np.array([1, -1])

def stop_left_momentum():
    return np.array([0.25, 1])

def stop_right_momentum():
    return np.array([0.25, -1])


def production_system(observation):
    if is_near_starting_point(observation):
        if not has_upwards_momentum(observation):
            if tilted_to_left(observation):
                return stabilize_right()
            elif tilted_to_right(observation):
                return stabilize_left()
            else:
                return np.array([0, 0])
        else:
            return np.array([0, 0])
    else:
        return np.array([0, 0])

    # if not is_near_landing_pad(observation):
    #     if not has_stable_orientation(observation):
    #         print('go up')
    #         return go_up()
    #     else:
    #         return np.array([0, 0])
    # else:
    #     return np.array([0, 0])

    # if moving_left(observation):
    #     if not tilted_to_right(observation):
    #         return stop_left_momentum()
    #     else:
    #         return stop_right_momentum()
    # elif moving_right(observation):
    #     if not tilted_to_left(observation):
    #         return stop_right_momentum()
    #     else:
    #         return stop_left_momentum()
    # else:
    #     return np.array([0, 0])  # Manter a posição atual

    # if has_stable_velocity(observation):
    #     if is_left_of_pad(observation):
    #         return go_right()
    #     elif is_right_of_pad(observation):
    #         return go_left()
    #     else:
    #         return np.array([0, 0])  # Manter a posição atual
    # else:
    #     return go_up()
    
def reactive_agent(observation):
    ##TODO: Implemente aqui o seu agente reativo
    ##Substitua a linha abaixo pela sua implementação
    #action = env.action_space.sample()
    #print('observação:',observation[3])
    action = production_system(observation)
    return action 
    
    
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
    
