import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib

class SmartLightingEnv(gym.Env):
    def __init__(self):
        super(SmartLightingEnv, self).__init__()
        self.ml_model = joblib.load('model_trafic.pkl')
        
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(low=0, high=10000, shape=(2,), dtype=np.float32)
        
        self.current_step = 0

    def step(self, action):
        self.current_step = (self.current_step + 1) % 24
        trafic_prezis = self.ml_model.predict([[self.current_step, 0, 20, 10]])[0] 
        
        reward = 0
        if action == 2: 
            if trafic_prezis > 2000: reward += 10  
            else: reward -= 5                      
        elif action == 0: # Lumină stinsă
            if trafic_prezis > 1000: reward -= 20 
            else: reward += 10                     
            
        done = False 
        obs = np.array([self.current_step, trafic_prezis], dtype=np.float32)
        
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        obs = np.array([0, 0], dtype=np.float32)
        return obs, {}

print("Mediul RL a fost definit cu succes!")
