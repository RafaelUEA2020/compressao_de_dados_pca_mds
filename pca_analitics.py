import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import os

def realizar_analise_completa(caminho_csv):
    # 1. Carregamento e Preparação Universal
    if not os.path.exists(caminho_csv):
        print(f"Erro: Arquivo {caminho_csv} não encontrado.")
        return

    df = pd.read_csv(caminho_csv)
    nome_base = os.path.splitext(os.path.basename(caminho_csv))[0]
    
    # Filtra apenas colunas numéricas e trata NaNs com a mediana
    df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df_clean = df_numeric.fillna(df_numeric.median())
    
    # Padronização (Z-score normalization)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # --- PARTE 1: RESULTADOS DEDICADOS AO PCA ---
    print(f"Iniciando PCA para {nome_base}...")
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Gráfico de Variância (Scree Plot)
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, align='center', label='Variância Individual')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Variância Acumulada', color='red')
    plt.title(f'PCA - Variância Explicada ({nome_base})')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Razão de Variância')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('pca_detalhado_variancia.png')
    
    # Projeção PCA 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c='skyblue', edgecolors='k', alpha=0.7)
    plt.title(f'Projeção PCA 2D ({nome_base})')
    plt.xlabel(f'PC1 ({exp_var[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({exp_var[1]*100:.1f}%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('pca_detalhado_projecao.png')

    # --- PARTE 2: RESULTADOS DEDICADOS AO MDS ---
    print(f"Iniciando MDS para {nome_base}...")
    mds = MDS(n_components=2, metric=True, random_state=42, normalized_stress='auto')
    mds_data = mds.fit_transform(scaled_data)
    stress = mds.stress_

    # Projeção MDS 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)
    plt.title(f'Projeção MDS 2D ({nome_base})\nStress Final: {stress:.4f}')
    plt.xlabel('Dimensão 1')
    plt.ylabel('Dimensão 2')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('mds_detalhado_projecao.png')

    # --- PARTE 3: GRÁFICO COMPARATIVO FINAL ---
    print("Gerando painel comparativo...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot PCA
    ax1.scatter(pca_data[:, 0], pca_data[:, 1], c='skyblue', edgecolors='k', alpha=0.7)
    ax1.set_title(f'PCA (Variância Total 2D: {(exp_var[0]+exp_var[1])*100:.2f}%)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot MDS
    ax2.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)
    ax2.set_title(f'MDS (Stress: {stress:.4f})')
    ax2.set_xlabel('Dimensão 1')
    ax2.set_ylabel('Dimensão 2')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'Análise Comparativa de Redução de Dimensionalidade: {nome_base}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparativo_final_pca_mds.png')
    
    print("\n[SUCESSO] Todos os gráficos foram gerados:")
    print("- pca_detalhado_variancia.png")
    print("- pca_detalhado_projecao.png")
    print("- mds_detalhado_projecao.png")
    print("- comparativo_final_pca_mds.png")

# Chamada da função (funciona com qualquer CSV)
realizar_analise_completa('dados/climate_change_dataset.csv')