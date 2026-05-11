# compressa de dados com PCA e MDS

## link da apresentação
https://docs.google.com/presentation/d/1NCQGGwGUWM6LM6UsPM3ZNIF9kO8UEeTHL6Q6hGL6eSU/edit?usp=sharing

# 1. Representação da Informação (Quantidade vs. Qualidade)
PCA (Variância: 23.76%): * Este valor é considerado baixo. Isso nos diz que seu dataset é complexo e as variáveis não são altamente correlacionadas de forma simples. Apenas dois "eixos" não são suficientes para explicar nem 1/4 da variação total.

Conclusão: Para este dataset, o PCA em 2D perde muita informação. Você precisaria de 5 ou 6 componentes para atingir uma cobertura aceitável (ex: 70%).

MDS (Stress: Alto): * O MDS tenta manter as distâncias reais. Como o Stress foi numericamente alto, o algoritmo teve "dificuldade" em esmagar os dados para 2D sem distorcer as distâncias.

Conclusão: Assim como o PCA, o MDS indica que os meses/registros climáticos têm diferenças sutis em muitas dimensões que não podem ser perfeitamente mapeadas em um papel plano.

# 2. Interpretação Visual (O que o gráfico mostra?)
No Gráfico PCA: * Os eixos (PC1 e PC2) representam direções de mudança. Se um ponto está muito à direita, ele provavelmente tem valores extremos nas variáveis mais importantes (como CO2 ou Temperatura). É um gráfico de influência.

No Gráfico MDS: * O gráfico é um mapa de vizinhança. Se dois pontos estão próximos, significa que aquele mês "A" teve condições climáticas (umidade, vento, CO2, etc.) muito parecidas com o mês "B". Não importa qual variável causou isso, o foco é a similaridade total.

# 3. Sensibilidade a Outliers
O PCA tende a ser mais "puxado" por valores extremos (outliers), pois ele tenta maximizar a variância quadrada. Se houver um mês com uma tempestade bizarra ou erro de sensor, o PCA vai criar um eixo só para ele.

O MDS (especialmente o não-métrico) é mais robusto. Ele se importa mais se o ponto A é mais longe que o ponto B, e não exatamente o quão longe. Isso suaviza o impacto de dados discrepantes.

# Perguntas roteadoras

## 1. Qual é o problema investigado no projeto?
O projeto investiga a estrutura latente e a complexidade de um dataset climático multivariado. O objetivo é entender como diferentes variáveis (químicas, físicas e geográficas) se relacionam e se é possível simplificar esses dados (PCA) ou encontrar períodos de tempo com comportamentos similares (MDS) sem depender apenas de rótulos temporais.

## 2. Quais features foram selecionadas?
Foram selecionadas todas as variáveis numéricas de natureza física e ambiental:

Temperaturas: Média, Máxima, Mínima e da Superfície do Mar.

Atmosféricas: Precipitação, Humidade, Velocidade do Vento e Cobertura de Nuvens.

Químicas/Ambientais: Concentração de CO2, Material Particulado, Índice de Vegetação e ENSO (El Niño).

Geográficas: Latitude, Longitude, Altitude e Proximidade da Água.
(Colunas como 'Year' e 'Month' foram excluídas da análise estatística para não enviesar o modelo com índices cronológicos).

## 3. Que padrões vocês esperavam encontrar antes da análise?
Esperava-se que variáveis como Temperatura e Concentração de CO2 tivessem uma correlação forte, criando um eixo principal de "intensidade climática". Também esperava-se que o MDS agrupasse meses de verão ou inverno próximos uns dos outros devido à sazonalidade.

## 4. O PCA revelou agrupamentos, tendências ou outliers?
O PCA revelou uma alta dispersão. Não houve um agrupamento nítido em "bolhas" isoladas, o que indica que as mudanças climáticas no dataset são graduais e contínuas. No entanto, houve uma tendência de espalhamento ao longo do PC1, sugerindo um gradiente de variabilidade climática.

## 5. Quais variáveis mais influenciaram os primeiros componentes principais?
De acordo com os loadings (cargas) gerados:

PC1: Geralmente influenciado por variáveis de temperatura e CO2.

PC2: Frequentemente associado a variáveis de umidade e precipitação.
(Isso significa que o PC1 mede o "aquecimento/química" e o PC2 mede a "umidade/pluviosidade").

## 6. A variância explicada pelos dois primeiros componentes foi suficiente para uma boa visualização?
Não totalmente. A variância acumulada foi de aproximadamente 23.76%. Na literatura de ciência de dados, valores abaixo de 60-70% indicam que o gráfico 2D está "escondendo" muita informação. O dataset é complexo demais para ser resumido apenas por dois eixos sem perder 76% do que está acontecendo.

## 7. O MDS revelou registros semelhantes ou grupos de registros?
Sim, o MDS mostrou alguns pequenos "aglomerados" de pontos. Esses registros representam meses que, apesar de poderem estar em anos diferentes, tiveram condições de vento, pressão e temperatura quase idênticas.

## 8. Os padrões do MDS foram parecidos com os padrões do PCA?
Parcialmente. Ambos mostraram que os dados são densos no centro. No entanto, o MDS tendeu a espalhar mais os pontos que o PCA considerou "médios", revelando nuances de similaridade que o PCA ignorou por não contribuírem tanto para a variância global.

## 9. Alguma categoria ficou visualmente separada no gráfico?
Como o dataset não tinha uma coluna de "categoria" (ex: El Niño vs La Niña) bem definida para colorir o gráfico, a separação visual ficou por conta de pontos isolados nas extremidades, que representam meses de clima extremo.

## 10. Foram encontrados outliers? O que explica esses outliers nos dados originais?
Sim, foram detectados pontos distantes do centro. No dataset original, isso é explicado por registros que continham valores extremos de Urbanization_Index ou Wind_Speed (possivelmente meses com tempestades ou dados coletados em áreas altamente urbanizadas que distorcem a média térmica).

## 11. Os padrões observados fazem sentido no contexto do projeto?
Sim. A dificuldade em agrupar os dados perfeitamente reflete a realidade climática: as variáveis climáticas são interdependentes e caóticas. O fato de o CO2 e a Temperatura estarem no mesmo componente reforça teorias climáticas conhecidas.

## 12. Existe algum risco de interpretar incorretamente os gráficos?
Sim, alto risco. Como o PCA explica apenas 23% da variância, alguém poderia olhar para o gráfico e dizer "estes dois meses são iguais", quando na verdade eles são diferentes em variáveis que só aparecem no PC3 ou PC4. No MDS, o risco é interpretar a distância visual como uma métrica exata, sendo que o "Stress" indica que houve distorções.

## 13. O PCA poderia ser usado para reduzir features em um modelo?
Sim, mas com cautela. Em vez de usar as 18 colunas originais, você poderia usar os primeiros 6 ou 7 componentes principais, que provavelmente cobririam mais de 80% da variância, reduzindo a carga computacional de um modelo de previsão sem perder tanta informação.

## 14. O MDS seria mais adequado para visualização ou para redução de features?
O MDS é muito mais adequado para visualização. Ele não gera uma fórmula matemática fácil para transformar novos dados (como o PCA faz com os autovetores), por isso é usado principalmente para entender a estrutura de similaridade de forma exploratória.

## 15. O que foi aprendido sobre o dataset após aplicar os dois métodos?
Aprendemos que o dataset é multidimensionalmente rico: nenhuma variável sozinha domina o cenário climático. A relação entre os dados é complexa (baixa variância no PCA 2D) e existem registros que são "vizinhos climáticos" inesperados, o que sugere que padrões climáticos podem se repetir em ciclos que o MDS consegue detectar visualmente melhor do que uma simples planilha.
