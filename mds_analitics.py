import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# 1. Carregar e Limpar os dados (mesmo processo do PCA)
df = pd.read_csv('dados\climate_change_dataset.csv')
df_numeric = df.drop(columns=['Year', 'Month'], errors='ignore')

for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Imputação pela mediana para evitar NaNs
df_clean = df_numeric.fillna(df_numeric.median())

# 2. Padronização
# O MDS métrico geralmente utiliza distâncias euclidianas. 
# Escalonar os dados é crucial para que uma variável (como CO2) não pese mais que outra (como Temp)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# 3. Aplicação do MDS
# n_components=2 para visualização em 2D
# metric=True realiza o MDS métrico (similar ao PCA, mas focado em distâncias)
mds = MDS(n_components=2, metric=True, random_state=42, normalized_stress='auto')
mds_data = mds.fit_transform(scaled_data)

# 4. Cálculo do Stress (Medida de qualidade do ajuste)
# O Stress indica o quanto da estrutura de distância original foi "perdida" na redução
stress = mds.stress_
print(f"Stress final do modelo: {stress:.2f}")

# 5. Visualização
plt.figure(figsize=(8, 6))
plt.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)

# Adicionando rótulos simples para identificar os pontos (opcional)
for i in range(len(mds_data)):
    if i % 5 == 0: # Rótula apenas alguns pontos para não poluir o gráfico
        plt.annotate(f"P{i}", (mds_data[i, 0], mds_data[i, 1]), fontsize=9, alpha=0.8)

plt.title('Escalonamento Multidimensional (MDS) - Clima')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('mds_plot.png')
plt.show()