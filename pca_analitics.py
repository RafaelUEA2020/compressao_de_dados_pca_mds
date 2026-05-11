import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Carregar os dados
df = pd.read_csv("dados\climate_change_dataset.csv")

# 2. Pré-processamento
# Removemos colunas temporais (Year, Month) para focar nas variáveis físicas
df_numeric = df.drop(columns=['Year', 'Month'], errors='ignore')

# Converte todas as colunas para numérico (trata strings como 'Unknown' como NaN)
for col in df_numeric.columns:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

# Preenche valores ausentes com a mediana de cada coluna
df_clean = df_numeric.fillna(df_numeric.median())

# 3. Padronização (Essencial para PCA)
# O PCA é sensível à escala. Deixamos todos os dados com média 0 e desvio padrão 1.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# 4. Aplicação do PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# 5. Visualização da Variância (Scree Plot)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(exp_var_pca)+1), exp_var_pca, alpha=0.5, align='center', label='Variância Individual')
plt.step(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, where='mid', label='Variância Acumulada')
plt.ylabel('Razão de Variância Explicada')
plt.xlabel('Componentes Principais')
plt.title('Scree Plot: Variância Explicada pelo PCA')
plt.legend(loc='best')
plt.savefig('pca_variance.png')

# 6. Visualização dos Dados em 2D (PC1 vs PC2)
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.7, edgecolors='k', c='teal')
plt.xlabel(f'PC1 ({exp_var_pca[0]:.2%} da variância)')
plt.ylabel(f'PC2 ({exp_var_pca[1]:.2%} da variância)')
plt.title('Projeção do Dataset de Mudanças Climáticas em 2D')
plt.grid(True)
plt.savefig('pca_2d.png')

# Salvar a importância de cada variável original nos componentes
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df_clean.columns)
loadings.to_csv('pca_loadings.csv')

print("Análise concluída. Gráficos 'pca_variance.png' e 'pca_2d.png' gerados.")