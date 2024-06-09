import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el Dataset
df = pd.read_csv('C:/INGENIERIA CIENCIA DE DATOS/ACTIVIDAD 4/Dataset  transporte_masivo.csv')

# 2. Preprocesar los Datos
# Convertir columnas de tiempo a minutos desde medianoche
def time_to_minutes(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 60 + m

df['hora_salida'] = df['hora_salida'].apply(time_to_minutes)
df['hora_llegada'] = df['hora_llegada'].apply(time_to_minutes)

# Convertir días de la semana a valores numéricos
dias_semana = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 'Viernes': 4}
df['dia_semana'] = df['dia_semana'].map(dias_semana)

# Codificar condiciones climáticas y eventos especiales como variables dummy
df = pd.get_dummies(df, columns=['condiciones_climaticas', 'eventos_especiales'])

# 3. Seleccionar Características
features = ['hora_salida', 'hora_llegada', 'origen_lat', 'origen_lon', 'destino_lat', 'destino_lon', 'num_pasajeros', 'dia_semana']
features += [col for col in df.columns if 'condiciones_climaticas_' in col or 'eventos_especiales_' in col]

X = df[features]

# 4. Normalizar los Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Aplicar K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Evaluar y Visualizar
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='origen_lon', y='origen_lat', hue='cluster', palette='viridis')
plt.title('Clusters de Orígenes de Viajes')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.show()
