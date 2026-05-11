import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import os

def realizar_analise_completa_com_tabela(caminho_csv):
    # 1. Carregamento e Preparação Universal
    if not os.path.exists(caminho_csv):
        print(f"Erro: Arquivo {caminho_csv} não encontrado.")
        return

    df = pd.read_csv(caminho_csv)
    nome_base = os.path.splitext(os.path.basename(caminho_csv))[0]
    
    # Filtra apenas colunas numéricas e trata NaNs com a mediana
    df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    df_clean = df_numeric.fillna(df_numeric.median())
    
    # --- NOVA ETAPA: EXPORTAR TABELA UTILIZADA ---
    nome_tabela = f"variaveis_numericas_{nome_base}.csv"
    df_clean.to_csv(nome_tabela, index=False)
    print(f"Tabela de variáveis numéricas exportada: {nome_tabela}")

    # Padronização (Crucial para PCA e MDS)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # --- PARTE 1: RESULTADOS DEDICADOS AO PCA ---
    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Gráfico de Variância
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(exp_var)+1), exp_var, alpha=0.5, label='Variância Individual')
    plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Variância Acumulada', color='red')
    plt.title(f'PCA - Variância Explicada ({nome_base})')
    plt.ylabel('Razão de Variância')
    plt.legend()
    plt.savefig('pca_detalhado_variancia.png')
    plt.close()

    # --- PARTE 2: RESULTADOS DEDICADOS AO MDS ---
    mds = MDS(n_components=2, metric=True, random_state=42, normalized_stress='auto')
    mds_data = mds.fit_transform(scaled_data)
    stress = mds.stress_

    # Projeção MDS
    plt.figure(figsize=(8, 6))
    plt.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)
    plt.title(f'MDS - Projeção 2D (Stress: {stress:.4f})')
    plt.savefig('mds_detalhado_projecao.png')
    plt.close()

    # --- PARTE 3: PAINEL COMPARATIVO LADO A LADO ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ax1.scatter(pca_data[:, 0], pca_data[:, 1], c='skyblue', edgecolors='k', alpha=0.7)
    ax1.set_title(f'PCA (Variância 2D: {(exp_var[0]+exp_var[1])*100:.2f}%)')

    ax2.scatter(mds_data[:, 0], mds_data[:, 1], c='salmon', edgecolors='k', alpha=0.7)
    ax2.set_title(f'MDS (Stress: {stress:.4f})')

    plt.suptitle(f'Painel Comparativo: {nome_base}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comparativo_final_pca_mds.png')
    plt.close()
    
    return nome_tabela

# Execução
realizar_analise_completa_com_tabela('dados/sinistros_transito_ocorrencia.csv')