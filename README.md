# Sistem inteligent de optimizare a iluminatului urban

Proiect de machine learning care simulează controlul iluminatului stradal în funcție
de traficul estimat. Un model de predicție a traficului (Random Forest) alimentează un
mediu de simulare în care un agent de reinforcement learning (DQN) învață când să
mențină lumina stinsă, la intensitate medie sau maximă, pentru a reduce consumul de
energie fără a sacrifica vizibilitatea în orele cu trafic intens.

## Cum funcționează
1. **Predicția traficului** (`antrenare.py`): un `RandomForestRegressor` (scikit-learn)
   este antrenat pe datasetul Metro Interstate Traffic Volume și salvat cu joblib
   (`model_trafic.pkl`).
2. **Mediul de simulare** (`mediu_rl.py`): mediu Gymnasium (`SmartLightingEnv`).
   - Observație: ora din zi și traficul prezis.
   - Acțiuni: 0 = lumină stinsă, 1 = intensitate medie, 2 = intensitate maximă.
   - Recompensă: se premiază lumina puternică la trafic intens și lumina stinsă la
     trafic redus; se penalizează lumina stinsă la trafic intens și lumina
     puternică la trafic redus.
3. **Antrenarea agentului** (`antrenare_rl.py`): agent DQN (stable-baselines3).
4. **Rularea** (`main.py`): simularea cu agentul antrenat.

## Tehnologii
Python, pandas, NumPy, scikit-learn, Gymnasium, Stable-Baselines3, joblib

## Instalare și rulare
  ```
pip install -r requirements.txt
python antrenare.py        # generează model_trafic.pkl
python antrenare_rl.py     # antrenează agentul DQN
python main.py             # rulează simularea
```


## Date
Metro Interstate Traffic Volume (UCI Machine Learning Repository), inclus în
`Metro_Interstate_Traffic_Volume.csv`.

## Context
Proiect realizat în cadrul atestatului de informatică, clasa a XII-a.

## Limitări și îmbunătățiri posibile
- Mediul folosește valori fixe pentru condițiile meteo la predicția traficului;
  ar putea folosi date reale sau variabile.
- Funcția de recompensă e simplă și nu include consumul real de energie; ar putea
  fi extinsă cu un cost pe nivel de intensitate.
- Episodul nu se termină niciodată; ar putea fi limitat la o zi simulată.
- Evaluare cantitativă a economiei de energie față de iluminatul constant.
