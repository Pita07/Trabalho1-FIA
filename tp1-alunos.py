import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
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
    return -0.2 < vy

def has_stable_orientation(observation):
    theta = observation[4]
    return abs(theta) < np.deg2rad(18)

def is_near_starting_point(observation):
    y = observation[1]
    return (1.45 > y > 1.2)

def is_near_landing_pad(observation):
    y = observation[1]
    return y < 0.4

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
    return np.array([0, 1])

def stabilize_left():
    return np.array([0, -1])

def stop_left_momentum():
    return np.array([0.25, 1])

def stop_right_momentum():
    return np.array([0.25, -1])

def adjust_right():
    return np.array([2.75, 0.65])

def adjust_left():
    return np.array([2.75, -0.65])

def production_system(observation):

    x = observation[0]
    y = observation[1]
    vx = observation[2]
    vy = observation[3]
    theta = observation[4]
    omega = observation[5]

    # --- Inclining control ---
    target_theta = (0.5 * x + 1.0 * vx)

    target_theta = np.clip(target_theta, -0.4, 0.4)

    # 0.05 error margin
    if target_theta + 0.05 < theta: #inclined to the left
        side = 1 #fire right engine to correct
    elif target_theta - 0.05 > theta: #inclined to the right
        side = -1 #fire left engine to correct
    else:
        side = 0 #no need to fire side engines

    # --- Height Control ---
    target_vy = abs(x) -0.4
    target_vy = np.clip(target_vy, -0.4, 0)

    if vy < target_vy: # if we're falling too fast
        main = 0.6 #fire main engine to slow down
    else:
        main = 0 #no need to fire main engine

    # --- Angular velocity control ---  
    if omega > 0.3: # if we're rotating too fast to the right
        side = 1 #fire left engine to correct
    elif omega < -0.3: # if we're rotating too fast to the left
        side = -1 #fire right engine to correct
    
    return np.array([main, side])

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
    
