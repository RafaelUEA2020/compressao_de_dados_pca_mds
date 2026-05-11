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
