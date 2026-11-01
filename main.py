import joblib
from stable_baselines3 import DQN
import numpy as np
import time

print("=" * 50)
print("SISTEM INTELIGENT DE ILUMINAT ADAPTIV (FAI) - EN 1320")
print("=" * 50)

print("[INFO] Se încarcă modelul de predicție trafic (Random Forest)...")
ml_model = joblib.load('model_trafic.pkl')

print("[INFO] Se încarcă agentul decizional (DQN Reinforcement Learning)...")
rl_agent = DQN.load("agent_iluminat_final")

print("\n[SUCCESS] Toate componentele AI au fost integrate cu succes.")
print("-" * 50)
print("SIMULARE CONTRUL ILUMINAT PE PARCURSUL A 24 DE ORE:")
print("-" * 50)

nume_decizii = {0: "STINS / MINIM (Economie)", 1: "50% INTENSITATE (Mediu)", 2: "100% INTENSITATE (Siguranță)"}

for ora in range(24):
    trafic_prezis = ml_model.predict([[ora, 0, 20, 10]])[0]
    
    stare = np.array([ora, trafic_prezis], dtype=np.float32)
    
    actiune, _ = rl_agent.predict(stare, deterministic=True)
    
    print(f"Ora {ora:02d}:00 | Trafic estimat: {int(trafic_prezis):4d} mașini | Decizie AI: {nume_decizii[int(actiune)]}")
    time.sleep(0.1) # Doar pentru un efect vizual de monitorizare în timp real

print("=" * 50)
print("Simulare încheiată cu succes!")
