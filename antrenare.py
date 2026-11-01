import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


print("Se încarcă datele din CSV...")
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

print("Preprocesare date...")
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek


features = ['hour', 'day_of_week', 'temp', 'clouds_all']
X = df[features]
y = df['traffic_volume']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Antrenare
print("Antrenare model Machine Learning (Random Forest)...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

#Evaluare
predicții = model.predict(X_test)
mae = mean_absolute_error(y_test, predicții)
print("-" * 30)
print(f"SUCCES! Eroare medie (MAE): {mae:.2f} mașini")
print("-" * 30)


joblib.dump(model, 'model_trafic.pkl')
print("Modelul a fost salvat ca 'model_trafic.pkl'")
