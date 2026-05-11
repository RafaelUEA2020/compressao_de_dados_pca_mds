import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import os

def aplicar_mds_generico(caminho_csv, n_componentes=2):
    # 1. Carregar os dados
    if not os.path.exists(caminho_csv):
        print(f"Erro: O arquivo {caminho_csv} não foi encontrado.")
        return
    
    df = pd.read_csv(caminho_csv)
    print(f"--- Processando arquivo: {caminho_csv} ---")
    print(f"Shape original: {df.shape}")

    # 2. Filtragem Automática de Dados Numéricos
    # Tentamos converter tudo o que for possível para numérico
    # Colunas que são puramente texto (IDs, Nomes, Datas complexas) serão ignoradas
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Removemos colunas que resultaram apenas em NaNs (eram textos não convertíveis)
    df_numeric = df_numeric.dropna(axis=1, how='all')
    
    # 3. Tratamento de Valores Ausentes (Imputação pela Mediana)
    if df_numeric.isnull().values.any():
        print("Valores ausentes detectados. Aplicando imputação pela mediana...")
        df_clean = df_numeric.fillna(df_numeric.median())
    else:
        df_clean = df_numeric

    # Verificação de segurança: existem colunas suficientes?
    if df_clean.shape[1] < n_componentes:
        print("Erro: O dataset não possui colunas numéricas suficientes para redução.")
        return

    # 4. Padronização
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # 5. Aplicação do MDS
    print(f"Executando MDS para {n_componentes} componentes...")
    mds = MDS(n_components=n_componentes, metric=True, random_state=42, normalized_stress='auto')
    mds_data = mds.fit_transform(scaled_data)

    # 6. Cálculo do Stress
    stress = mds.stress_
    print(f"Stress final do modelo: {stress:.4f}")

    # 7. Visualização
    plt.figure(figsize=(10, 7))
    
    # Se houver uma coluna de texto original (ex: nomes de cidades ou categorias), 
    # podemos usá-la para legenda, caso contrário, usamos o índice
    scatter = plt.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)

    # Lógica de anotação inteligente (limita a 20 pontos para não poluir)
    passo = max(1, len(mds_data) // 20)
    for i in range(0, len(mds_data), passo):
        plt.annotate(f"Id:{df.index[i]}", (mds_data[i, 0], mds_data[i, 1]), 
                     fontsize=8, alpha=0.7, xytext=(5,5), textcoords='offset points')

    plt.title(f'Visualização MDS - {os.path.basename(caminho_csv)}')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    nome_saida = f"mds_result_{os.path.basename(caminho_csv)}.png"
    plt.savefig(nome_saida)
    print(f"Gráfico salvo como: {nome_saida}")
    plt.show()

# --- Exemplo de uso ---
# Basta substituir pelo nome do seu arquivo atual
aplicar_mds_generico('dados/sinistro_transito_ocorrencia.csv')