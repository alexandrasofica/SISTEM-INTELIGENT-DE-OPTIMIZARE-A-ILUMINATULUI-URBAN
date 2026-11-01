from stable_baselines3 import DQN
from mediu_rl import SmartLightingEnv 

env = SmartLightingEnv()

model = DQN("MlpPolicy", env, verbose=1)

print("Agentul învață acum... Te rog așteaptă câteva minute.")
model.learn(total_timesteps=10000)

model.save("agent_iluminat_final")
print("Antrenare finalizată! Agentul a fost salvat.")


obs, _ = env.reset()
for i in range(5):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = env.step(action)
    print(f"Ora: {int(obs[0])}, Trafic: {int(obs[1])}, Decizie Agent: {action}")
