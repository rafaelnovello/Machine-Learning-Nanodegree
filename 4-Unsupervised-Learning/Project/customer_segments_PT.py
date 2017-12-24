
# coding: utf-8

# # Nanodegree Engenheiro de Machine Learning
# ## Unsupervised Learning
# ## Project 3: Creating Customer Segments

# Bem-vindo ao terceiro projeto do Nanodegree Engenheiro de Machine Learning! Neste Notebook, alguns modelos de código já foram fornecidos e será seu trabalho implementar funcionalidades adicionais necessárias para completar seu projeto com êxito. Seções que começam com **'Implementação'** no cabeçalho indicam que os blocos de código seguintes vão precisar de funcionalidades adicionais que você deve fornecer. As instruções serão fornecidas para cada seção e as especificações da implementação são marcados no bloco de código com um `'TODO'`. Leia as instruções atentamente!
# 
# Além de implementar códigos, há perguntas que você deve responder relacionadas ao projeto e a sua implementação. Cada seção na qual você responderá uma questão está precedida de um cabeçalho **'Questão X'**. Leia atentamente cada questão e forneça respostas completas nos boxes seguintes que começam com **'Resposta:'**. O envio do seu projeto será avaliado baseado nas suas respostas para cada uma das questões e na implementação que você forneceu.  
# 
# >**Nota:** Células de código e Markdown podem ser executadas utilizando o atalho do teclado **Shift+Enter**. Além disso, células de Markdown podem ser editadas ao dar duplo clique na célula para entrar no modo de edição.

# ## Começando
# 
# Neste projeto, você irá analisar o conjunto de dados de montantes de despesas anuais de vários clientes (reportados em *unidades monetárias*) de diversas categorias de produtos para estrutura interna. Um objetivo deste projeto é melhor descrever a variação de diferentes tipos de clientes que um distribuidor de atacado interage. Isso dará ao distribuidor discernimento sobre como melhor estruturar seu serviço de entrega de acordo com as necessidades de cada cliente.
# 
# O conjunto de dados deste projeto pode ser encontrado no [Repositório de Machine Learning da UCI](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). Para efeitos de projeto, os atributos `'Channel'` e `'Region'` serão excluídos da análise – que focará então nas seis categorias de produtos registrados para clientes.
# 
# Execute o bloco de código abaixo para carregar o conjunto de dados de clientes da distribuidora, junto com algumas das bibliotecas de Python necessárias exigidos para este projeto. Você saberá que o conjunto de dados carregou com êxito se o tamanho do conjunto de dados for reportado.

# In[1]:


# Importe as bibliotecas necessárias para este projeto
import numpy as np
import pandas as pd
import renders as rs
from IPython.display import display # Allows the use of display() for DataFrames

# Mostre matplotlib no corpo do texto (bem formatado no Notebook)
get_ipython().magic(u'matplotlib inline')

# Carregue o conjunto de dados dos clientes da distribuidora de atacado
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# ## Observando os Dados
# Nesta seção, você vai começar a explorar os dados através de visualizações e códigos para entender como cada atributo é relacionado a outros. Você vai observar descrições estatísticas do conjunto de dados, considerando a relevância de cada atributo, e selecionando alguns exemplos de pontos de dados do conjunto de dados que você vai seguir no decorrer do curso deste projeto.
# 
# Execute o bloco de código abaixo para observar as descrições estatísticas sobre o conjunto de dados. Note que o conjunto é compostos de seis categorias importantes de produtos: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'** e **'Delicatessen'**. Considere o que cada categoria representa em termos os produtos que você poderia comprar.

# In[2]:


# Mostre a descrição do conjunto de dados
display(data.describe())


# ### Implementação: Selecionando Amostras
# Para melhor compreensão da clientela e como seus dados vão se transformar no decorrer da análise, é melhor selecionar algumas amostras de dados de pontos e explorá-los com mais detalhes. No bloco de código abaixo, adicione **três** índices de sua escolha para a lista de `indices` que irá representar os clientes que serão acompanhados. Sugerimos que você tente diferentes conjuntos de amostras até obter clientes que variam significativamente entre si.

# In[3]:


indices = [5,10,75]

# Crie um DataFrame das amostras escolhidas
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
samples.head(15)


# ### Questão 1
# Considere que a compra total de cada categoria de produto e a descrição estatística do conjunto de dados abaixo para a sua amostra de clientes.  
# *Que tipo de estabelecimento (de cliente) cada uma das três amostras que você escolheu representa?*  
# **Dica:** Exemplos de estabelecimentos incluem lugares como mercados, cafés e varejistas, entre outros. Evite utilizar nomes para esses padrões, como dizer *"McDonalds"* ao descrever uma amostra de cliente de restaurante.

# **Resposta:**
# 
# - Indice 5 (0): Restaurante - Esta amostra tem uma grande quantidade de alimentos frescos, leite e uma pequena quantidade de alimentos congelados, dando a entender que a comida é feita para consumo imediato, então é provável que seja um restaurante.
# 
# - Indice 10 (1): Mercearia - Esta amostra tem valores altos em todas as features, com uma grande proporção de mantimentos. Isso sugere que é uma mercearia, talvez um supermercado.
# 
# - Indice 75 (2): Fornecedor - Comidas predominantemente frescas, com um valor muito grande (bem acima do percentil 75 na feature Fresh). Isso sugere um varejista ou um produtor em massa ou um mercado de alimentos frescos.

# ### Implementação: Relevância do Atributo
# Um pensamento interessante a se considerar é se um (ou mais) das seis categorias de produto são na verdade relevantes para entender a compra do cliente. Dito isso, é possível determinar se o cliente que comprou certa quantidade de uma categoria de produto vai necessariamente comprar outra quantidade proporcional de outra categoria de produtos? Nós podemos determinar facilmente ao treinar uma aprendizagem não supervisionada de regressão em um conjunto de dados com um atributo removido e então pontuar quão bem o modelo pode prever o atributo removido.
# 
# No bloco de código abaixo, você precisará implementar o seguinte:
#  - Atribuir `new_data` a uma cópia dos dados ao remover o atributo da sua escolha utilizando a função `DataFrame.drop`.
#  - Utilizar `sklearn.cross_validation.train_test_split` para dividir o conjunto de dados em conjuntos de treinamento e teste.
#    - Utilizar o atributo removido como seu rótulo alvo. Estabelecer um `test_size` de `0.25` e estebeleça um `random_state`.
#  - Importar uma árvore de decisão regressora, estabelecer um `random_state` e ajustar o aprendiz nos dados de treinamento.
#  - Reportar a pontuação da previsão do conjunto de teste utilizando a função regressora `score`.

# In[4]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split as tt_split

# TODO: Fazer uma cópia do DataFrame utilizando a função 'drop' para soltar o atributo dado
new_data = data.drop('Grocery', axis=1)

# TODO: Dividir os dados em conjuntos de treinamento e teste utilizando o atributo dado como o alvo
X_train, X_test, y_train, y_test = tt_split(new_data, data['Grocery'], test_size=.25, random_state=42)

# TODO: Criar um árvore de decisão regressora e ajustá-la ao conjunto de treinamento
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)


# TODO: Reportar a pontuação da previsão utilizando o conjunto de teste
score = regressor.score(X_test, y_test)
score


# ### Questão 2
# *Qual atributo você tentou prever? Qual foi a pontuação da previsão reportada? Esse atributo é necessário para identificar os hábitos de compra dos clientes?*  
# **Dica:** O coeficiente de determinação, `R^2`, é pontuado entre 0 e 1, sendo 1 o ajuste perfeito. Um `R^2` negativo indica que o modelo falhou em ajustar os dados.

# **Resposta:**
# 
# Foi usado o atributo Grocery e a pontuação obtida foi de 69%. Eu acredito que este atributo é sim necessário para identificar os hábitos de compra dos clientes por se tratar de um perfil de compra mais essencial, como identificado na questão 1 são mercados, mercearias e afins onde são vendidos produtos mais importantes e essenciais no dia-a-dia.

# ### Visualizando a Distribuição de Atributos
# Para entender melhor o conjunto de dados, você pode construir uma matriz de dispersão de cada um dos seis atributos dos produtos presentes nos dados. Se você perceber que o atributo que você tentou prever acima é relevante para identificar um cliente específico, então a matriz de dispersão abaixo pode não mostrar nenhuma relação entre o atributo e os outros. Da mesma forma, se você acredita que o atributo não é relevante para identificar um cliente específico, a matriz de dispersão pode mostrar uma relação entre aquele e outros atributos dos dados. Execute o bloco de código abaixo para produzir uma matriz de dispersão.

# In[5]:


# Produza uma matriz de dispersão para cada um dos pares de atributos dos dados
pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Questão 3
# *Há algum par de atributos que mostra algum grau de correlação? Isso confirma ou nega a suspeita sobre relevância do atributo que você tentou prever? Como os dados desses atributos são distribuídos?*  
# **Dica:** Os dados são distribuídos normalmente? Onde a maioria dos pontos de dados está? 

# **Resposta:**
# 
# Aparentemente, Grocery e Detergents_paper apresentão a maior correlação, mas Milk e Grocery e Milk e Detergents_paper também apresentão algum grau de correlação.
# 
# Esta correlação confirma minha suspeita em relação a importancia do atributo Grocery já que o mesmo aparenta correlação com outros atributos do dataset.
# 
# Os dados entre Grocery e Detergents_paper são distribuidos de forma bem linear, mostrando uma correlação direta entre eles.

# ## Pré-processamento de Dados
# Nesta seção, você irá pré-processar os dados para criar uma melhor representação dos clientes ao executar um escalonamento dos dados e detectando os discrepantes. Pré-processar os dados é geralmente um passo fundamental para assegurar que os resultados obtidos na análise são importantes e significativos.

# ### Implementação: Escalonando Atributos
# Se os dados não são distribuídos normalmente, especialmente se a média e a mediana variam significativamente (indicando um grande desvio), é quase sempre [apropriado] ](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) aplicar um escalonamento não linear – particularmente para dados financeiros. Uma maneira de conseguir escalonar dessa forma é utilizando o [ teste Box-Cox](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), que calcula o melhor poder de transformação dos dados, que reduzem o desvio. Uma abordagem simplificada que pode funcionar na maioria dos casos seria aplicar o algoritmo natural.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribua uma cópia dos dados para o `log_data` depois de aplicar um algoritmo de escalonamento. Utilize a função `np.log` para isso.
#  - Atribua uma cópia da amostra do dados para o `log_samples` depois de aplicar um algoritmo de escalonamento. Novamente, utilize o `np.log`.

# In[6]:


# TODO: Escalone os dados utilizando o algoritmo natural
log_data = np.log(data)

# TODO: Escalone a amostra de dados utilizando o algoritmo natural
log_samples = np.log(samples)

# Produza uma matriz de dispersão para cada par de atributos novos-transformados
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# ### Observação
# Após aplicar o algoritmo natural para o escalonamento dos dados, a distribuição para cada atributo deve parecer mais normalizado. Para muitos pares de atributos, você vai precisar identificar anteriormente como sendo correlacionados, observe aqui se essa correlação ainda está presente (e se está mais forte ou mais fraca que antes).
# 
# Execute o código abaixo para ver como a amostra de dados mudou depois do algoritmo natural ter sido aplicado a ela.

# In[7]:


# Mostre a amostra dados log-transformada
log_samples.head(20)


# ### Implementação: Detecção de Discrepantes
# Identificar dados discrepantes é extremamente importante no passo de pré-processamento de dados de qualquer análise. A presença de discrepantes podem enviesar resultados que levam em consideração os pontos de dados. Há muitas "regras básicas" que constituem um discrepante em um conjunto de dados. Aqui usaremos [o Método Turco para identificar discrepantes](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): Um *passo do discrepante* é calculado 1,5 vezes a variação interquartil (IQR). Um ponto de dados com um atributo que está além de um passo de um discrepante do IQR para aquele atributo, ele é considerado anormal.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribuir o valor do 25º percentil do atributo dado para o `Q1`. Utilizar `np.percentile` para isso.
#  - Atribuir o valor do 75º percentil do atributo dado para o `Q3`. Novamente, utilizar `np.percentile`.
#  - Atribuir o cálculo de um passo do discrepante do atributo dado para o `step`.
#  - Remover opcionalmentos os pontos de dados do conjunto de dados ao adicionar índices à lista de `outliers`.
# 
# **NOTA:** Se você escolheu remover qualquer discrepante, tenha certeza que a amostra de dados não contém nenhum desses pontos!  
#  Uma vez que você executou essa implementação, o conjunto de dado será armazenado na variável `good_data`!Once you have performed this implementation, the dataset will be stored in the variable .

# In[8]:


outliers = []

# Para cada atributo encontre os pontos de dados com máximos valores altos e baixos
for feature in log_data.keys():
    
    # TODO: Calcule Q1 (25º percentil dos dados) para o atributo dado
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calcule Q3 (75º percentil dos dados) para o atributo dado
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Utilize a amplitude interquartil para calcular o passo do discrepante (1,5 vezes a variação interquartil)
    step = (Q3 - Q1) * 1.5
    
    # Mostre os discrepantes
    print "Data points considered outliers for the feature '{}':".format(feature)
    feature_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(feature_outliers)
    
    # OPCIONAL: Selecione os índices dos pontos de dados que você deseja remover
    outliers += feature_outliers.index.tolist()

# Remova os discrepantes, caso nenhum tenha sido especificado
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# ### Questão 4
# *Há alguns pontos de dado considerados discrepantes de mais de um atributo baseado na definição acima? Esses pontos de dados deveriam ser removidos do conjunto? Se qualquer ponto de dados foi adicionado na lista `outliers` para ser removido, explique por quê.* 

# **Resposta:**
# 
# Vários datapoints foram outliers para mais de uma feature
# 
# - 154: Outlier em Delicatessen, Milk e Grocery.
# - 128: Outlier em Delicatessen ae Fresh.
# - 75:  Outlier em Detergents_Paper e Grocery.
# - 66:  Outlier em Delicatessen e Fresh
# - 65:  Outlier em Frozen e Fresh
# 
# Todos os outliers são valores muito menores que a média de cada feature ou mesmo que o IQR. Os datapoints mencionados acima chamam mais a atenção por serem outliers em várias features ao mesmo tempo.
# 
# Como os outliers são muito menores que a média do dataset acredito ser interessante remove-los para não interferirem nos resultados.
# 
# Desta forma, todos os outliers identificados foram adicionados para remoção por serem realmente muito distastantes da média de cada feature.

# ## Transformação de Atributo
# Nesta seção, você irá utilizar a análise de componentes principais (PCA) para elaborar conclusões sobre a estrutura subjacente de dados de clientes do atacado Dado que ao utilizar a PCA em conjunto de dados calcula as dimensões que melhor maximizam a variância, nós iremos encontrar quais combinações de componentes de atributos melhor descrevem os consumidores.

# ### Implementação: PCA
# 
# Agora que os dados foram escalonados em uma distribuição normal e qualquer discrepante necessário foi removido, podemos aplicar a PCA na `good_data` para descobrir qual dimensão dos dados melhor maximizam a variância dos atributos envolvidos. Além de descobrir essas dimensões, a PCA também irá reportar a *razão da variância explicada* de cada dimensão – quanta variância dentro dos dados é explicada pela dimensão sozinha. Note que o componente (dimensão) da PCA pode ser considerado como um novo "feature" do espaço, entretanto, ele é uma composição do atributo original presente nos dados.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Importar o `sklearn.decomposition.PCA` e atribuir os resultados de ajuste da PCA em seis dimensões com o `good_data` para o `pca`.
#  - Aplicar a transformação da PCA na amostra de log-data `log_samples` utilizando `pca.transform`, e atribuir os resultados para o `pca_samples`.

# In[9]:


from sklearn.decomposition import PCA

# TODO: Aplique a PCA ao ajustar os bons dados com o mesmo número de dimensões como atributos
pca = PCA(n_components=6)
pca.fit(good_data)

# TODO: Transforme a amostra de data-log utilizando o ajuste da PCA acima
pca_samples = pca.transform(log_samples)

# Gere o plot dos resultados da PCA
pca_results = rs.pca_results(good_data, pca)


# ### Questão 5
# *Quanta variância nos dados é explicado * ***no total*** * pelo primeiro e segundo principal componente? E os quatro primeiros principais componentes? Utilizando a visualização fornecida acima, discuta quais das quatro primeiras dimensões melhor representa em termos de despesas dos clientes.*  
# **Dica:** Uma melhora positiva dentro de uma dimensão específica corresponde a uma *melhora* do atributos de *pesos-positivos* e uma *piora* dos atributos de *pesos-negativos*. A razão de melhora ou piora é baseada nos pesos de atributos individuais.

# **Resposta:**
# 
# Componentes 1 a 2:
# - 1º Comp: 49,9%
# - 2º Comp: 22,6%
# - Total:   72,5%
# 
# Componentes 1 a 4:
# - 3º Comp: 10,5%
# - 4º Comp:  9,8%
# - Total    92,8%
# 
# Cada componente representa diferentes setores de gastos dos clientes:
# 
# - O primeiro componente representa uma grande variedade, principalmente Detergents_Paper, mas também fornece informações para Milk e Grocery. Porem, representa mal as categorias Fresh e Frozen e precisa do 2º componente para ajudar. Ele poderia representar a categoria de gastos com "conveniência" ou "supermercado".
# 
# - O segundo componente representa melhor os recursos Fresh, Frozen e Delicatessen enquanto representa mal os outros recursos. Ele pode representar os clientes que estão na indústria de hotelaria ou restaurantes.
# 
# - O terceiro componente representa Fresh e Detergents_Paper. Ele poderia representar pequenas lojas, com itens de conveniência e pequenas quantidades de mantimentos.
# 
# - O quarto componente representa Frozen e Detergents_Paper. Ele poderia representar compradores em massa de produtos congelados, como importadores de peixe.

# ### Observação
# Execute o código abaixo para ver como a amostra de log transformado mudou depois de receber a transformação da PCA aplicada a ele em seis dimensões. Observe o valor numérico para as quatro primeiras dimensões para os pontos da amostra. Considere se isso for consistente com sua interpretação inicial dos pontos da amostra.

# In[10]:


# Exiba a amostra de log-data depois de aplicada a tranformação da PCA
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementação: Redução da Dimensionalidade
# Ao utilizar um componente principal de análise, um dos objetivos principais é reduzir a dimensionalidade dos dados – na realidade, reduzindo a complexidade do problema. Redução de dimensionalidade tem um custo: Poucas dimensões utilizadas implicam em menor variância total dos dados que estão sendo explicados. Por causo disso, a *taxa de variância explicada cumulativa* é extremamente importante para saber como várias dimensões são necessárias para o problema. Além disso, se uma quantidade significativa de variância é explicada por apenas duas ou três dimensões, os dados reduzidos podem ser visualizados depois.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Atribuir os resultados de ajuste da PCA em duas dimensões com o `good_data` para o `pca`.
#  - Atribuir a tranformação da PCA do `good_data` utilizando `pca.transform`, e atribuir os resultados para `reduced_data`.
#  - Aplicar a transformação da PCA da amostra do log-data `log_samples` utilizando `pca.transform`, e atribuindo os resultados ao `pca_samples`.

# In[11]:


# TODO: Aplique o PCA ao ajusta os bons dados com apenas duas dimensões
pca = PCA(n_components=2)
pca.fit(good_data)

# TODO: Transforme os bons dados utilizando o ajuste do PCA acima
reduced_data = pca.transform(good_data)

# TODO: Transforme a amostre de log-data utilizando o ajuste de PCA acima
pca_samples = pca.transform(log_samples)

# Crie o DataFrame para os dados reduzidos
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
rs.pca_results(good_data, pca)


# ### Observação
# Execute o código abaixo para ver como a amostra de dados do log-transformado mudou depois de receber a transformação do PCA aplicada a ele em seis dimensões. Observe o valor numérico para as quatro primeiras dimensões para os pontos da amostra. Considere se isso for consistente com sua interpretação inicial dos pontos da amostra.

# In[12]:


# Exiba a amostra de log-data depois de aplicada a transformação da PCA em duas dimensões
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Clustering
# 
# Nesta seção, você irá escolher utilizar entre o algoritmo de clustering K-Means ou o algoritmo de clustering do Modelo de Mistura Gaussiano para identificar as várias segmentações de clientes escondidos nos dados. Então você irá recuperar pontos de dados específicos do cluster para entender seus significados ao transformá-los de volta em suas dimensões e escalas originais. 

# ### Questão 6
# *Quais são as vantagens de utilizar o algoritmo de clustering K-Means? Quais são as vantagens de utilizar o algoritmo de clustering do Modelo de Mistura Gaussiano? Dadas as suas observações até agora sobre os dados de clientes da distribuidora, qual dos dois algoritmos você irá utilizar e por quê.*

# **Resposta:**
# 
# - Um algoritmo de agrupamento K-means tem menos parâmetros, como resultado, é muito mais rápido e adequado para situações com muitos dados e onde os clusters são claramente separados. Os pontos de dados pertencem de forma rígida a um cluster ou outro.
# 
# - Um Modelo de Mistura Gaussiano tem mais parâmetros e é um método de "agrupamento suave". Ao usar distribuições gaussianas e probabilidades, os pontos de dados podem ser atribuídos a múltiplos clusters ao mesmo tempo. Além disso, ele pode ser usado para prever probabilidades de eventos em vez de características rígidas.
# 
# Dada a natureza deste problema e as análises e visualizações anteriores acredito ser mais lógico adotar um Modelo Gaussiano de Mistura neste caso.

# ### Implementação: Criando Clusters
# Dependendo do problema, o número de clusters que você espera que estejam nos dados podem já ser conhecidos. Quando um número de clusters não é conhecido *a priori*, não há garantia que um dado número de clusters melhor segmenta os dados, já que não é claro quais estruturas existem nos dados – se existem. Entretanto, podemos quantificar a "eficiência" de um clustering ao calcular o *coeficiente de silhueta* de cada ponto de dados. O [coeficiente de silhueta](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) para um ponto de dado mede quão similar ele é do seu cluster atribuído, de -1 (não similar) a 1 (similar). Calcular a *média* do coeficiente de silhueta fornece um método de pontuação simples de um dado clustering.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Ajustar um algoritmo de clustering para o `reduced_data` e atribui-lo ao `clusterer`.
#  - Prever o cluster para cada ponto de dado no `reduced_data` utilizando o `clusterer.predict` e atribuindo eles ao `preds`.
#  - Encontrar os centros do cluster utilizando o atributo respectivo do algoritmo e atribuindo eles ao `centers`.
#  - Prever o cluster para cada amostra de pontos de dado no `pca_samples` e atribuindo eles ao `sample_preds`.
#  - Importar sklearn.metrics.silhouette_score e calcular o coeficiente de silhueta do `reduced_data` contra o do `preds`.
#    - Atribuir o coeficiente de silhueta para o `score` e imprimir o resultado.

# In[13]:


from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

for i in range(2, 11):
    # TODO: Aplique o algoritmo de clustering de sua escolha aos dados reduzidos 
    clusterer = GMM(n_components=i, random_state=42)
    clusterer.fit(reduced_data)

    # TODO: Preveja o cluster para cada ponto de dado
    preds = clusterer.predict(reduced_data)

    # TODO: Ache os centros do cluster
    centers = clusterer.means_

    # TODO: Preveja o cluster para cada amostra de pontos de dado transformados
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calcule a média do coeficiente de silhueta para o número de clusters escolhidos
    score = silhouette_score(reduced_data, preds)
    print i, score


# ### Questão 7
# *Reporte o coeficiente de silhueta para vários números de cluster que você tentou. Dentre eles, qual a quantidade de clusters que tem a melhor pontuação de silhueta?* 

# **Resposta:**
# 
# O coeficiente de silhueta para até 10 clusters esta acima. O melhor resultado foi obtido com 2 clusters com a pontuação de 0.44.

# ### Visualização de Cluster
# Uma vez que você escolheu o número ótimo de clusters para seu algoritmo de clustering utilizando o método de pontuação acima, agora você pode visualizar os resultados ao executar o bloco de código abaixo. Note que, para propósitos de experimentação, é de bom tom que você ajuste o número de clusters para o seu algoritmo de cluster para ver várias visualizações. A visualização final fornecida deve, entretanto, corresponder com o número ótimo de clusters. 

# In[14]:


clusterer = GMM(n_components=2, random_state=42)
clusterer.fit(reduced_data)

# TODO: Preveja o cluster para cada ponto de dado
preds = clusterer.predict(reduced_data)

# TODO: Ache os centros do cluster
centers = clusterer.means_


# Mostre os resultados do clustering da implementação
rs.cluster_results(reduced_data, preds, centers, pca_samples)


# ### Implementação: Recuperação de Dados
# Cada cluster apresentado na visualização acima tem um ponto central. Esses centros (ou médias) não são especificamente pontos de dados não específicos dos dados, em vez disso, são *as médias* de todos os pontos estimados em seus respectivos clusters. Para o problema de criar segmentações de clientes, o ponto central do cluster corresponde *a média dos clientes daquele segmento*. Já que os dados foram atualmente reduzidos em dimensões e escalas por um algoritmo, nós podemos recuperar a despesa representativa do cliente desses pontos de dados ao aplicar transformações inversas.
# 
# No bloco de código abaixo, você vai precisar implementar o seguinte:
#  - Aplicar a transformação inversa para o `centers` utilizando o `pca.inverse_transform`, e atribuir novos centros para o `log_centers`.
#  - Aplicar a função inversa do `np.log` para o `log_centers` utilizando `np.exp`, e atribuir os verdadeiros centros para o `true_centers`.
# 

# In[15]:


# TODO: Transforme inversamento os centros
log_centers = pca.inverse_transform(centers)

# TODO: Exponencie os centros
true_centers = np.exp(log_centers)

# Mostre os verdadeiros centros
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Questão 8
# Considere o gasto total de compra de cada categoria de produto para os pontos de dados representativos acima e reporte a descrição estatística do conjunto de dados no começo do projeto. Qual conjunto de estabelecimentos cada segmentação de clientes representa?*  
# **Dica:** Um cliente que é atribuído ao `'Cluster X'` deve se identificar melhor com os estabelecimentos representados pelo conjunto de atributos do `'Segment X'`.

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap((true_centers-data.mean())/data.std(ddof=0), square=True, annot=True, cbar=False);


# In[17]:


plt.figure()
plt.axes().set_title("Segment 0")
sns.barplot(x=true_centers.columns.values,y=true_centers.iloc[0].values)

plt.figure()
plt.axes().set_title("Segment 1")
sns.barplot(x=true_centers.columns.values,y=true_centers.iloc[1].values)


# **Resposta:**
# 
# - Cluster 1: provavelmente representa cafés e/ou restaurantes que servem comida fresca devido ao forte peso sobre a categoria Fresh. É consistente com a previsão original para como um restaurante deveria parecer.
# 
# - Cluster 2: as quantidades de Grocery e Milk são predominantes, os valores destes segmentos neste cluster excedem as medias observadas na seção de exploração dos dados, o que sugere que são distribuidores a granel ou grandes revendedores, como supermercados.

# ### Questão 9
# *Para cada amostra de ponto, qual segmento de cliente da* ***Questão 8*** *é melhor representado? As previsões para cada amostra de ponto são consistentes com isso?*
# 
# Execute o bloco de códigos abaixo para saber a previsão de segmento para cada amostra de ponto.

# In[18]:


# Mostre as previsões
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


# **Resposta:**
# 
# - Indice 5 (0)
#  - Avaliação inicial: Restaurante devido a combinação Milk e Fresh
#  - Avaliação do modelo: Supermercado
#  - Comentário: Minha interpretação da combinação Fresh + Milk foi de que se tratava de um restaurante mas o modelo avaliou de forma diferente.
# 
# - Indice 10 (1)
#  - Avaliação inicial: Mercearia devido a Grocery e Milk
#  - Avaliação do modelo: Supermercado
#  - Comentário: Eu entendi a predominância de Grocery como sendo uma caracteristica de supermercados e neste caso acertei!
#  
# - Indice 75 (0)
#  - Avaliação inicial: Fornecedor devido a Fresh
#  - Avaliação do modelo: Restaurante
#  - Comentário: Eu entendi a predominância de Fresh como sendo uma caracteristica de fornecedores e o modelo avaliou como restaurante.
#  
# 
# Esta comparação pode sugerir que o modelo não performou bem ou que minha avaliação dos exemplos foi muito distorcida, mas acredito que seja apenas uma diferença na interpretação. O modelo avaliou que um cliente com uma variedade de características fortes como Fresh, Milk, Grocery, Frozen sugere que seja o Cluster 2 (supermercado ou distribuidor). Clientes com um foco particular em uma única caracteristica (Fresh)  são considerados Cluster 1 (restaurantes). Isso realmente parece uma interpretação válida que poderia ser feita inicialmente.

# ## Conclusão

# Nesta seção final, você irá investigar maneiras de fazer uso dos dados que estão em clusters. Primeiro você vai considerar quais são os diferentes grupos de clientes, a **segmentação de clientes**, que pode ser afetada diferentemente por um esquema de entrega específico. Depois, você vai considerar como dar um rótulo para cada cliente (qual *segmento* aquele cliente pertence), podendo fornecer atributos adicionais sobre os dados do cliente. Por último, você vai comparar a **segmentação de clientes** com uma variável escondida nos dados, para ver se o cluster identificou certos tipos de relação.

# ### Questão 10
# Empresas sempre irão executar os [testes A/B](https://en.wikipedia.org/wiki/A/B_testing) ao fazer pequenas mudanças em seus produtos ou serviços para determinar se ao fazer aquela mudança, ela afetará seus clientes de maneira positiva ou negativa. O distribuidor de atacado está considerando mudar seu serviço de entrega de atuais 5 dias por semana para 3 dias na semana. Mas o distribuidor apenas fará essa mudança no sistema de entrega para os clientes que reagirem positivamente. *Como o distribuidor de atacado pode utilizar a segmentação de clientes para determinar quais clientes, se há algum, que serão alcançados positivamente à mudança no serviço de entrega?*  
# **Dica:** Podemos supor que as mudanças afetam todos os clientes igualmente? Como podemos determinar quais grupos de clientes são os mais afetados?

# **Resposta:**
# 
# - O modelo estabeleceu dois tipos principais de clientes, o Cluster 2 supermercados e/ou distribuidores a granel e o Cluster 1 restaurantes que armazenam alimentos frescos.
# 
# - É possível que os clientes do Cluster 1 que serve alimentos frescos vão querer entrega de  5 dias por semana para manter a comida fresca
# 
# - O Cluster 2 pode ser mais flexível, já que eles compram uma variedade maior de produtos perecíveis e não perecíveis, portanto, não necessitam de uma entrega diária.
# 
# Com isso em mente, a empresa poderia executar testes A/B e avaliar. Ao escolher um subconjunto de clientes de cada Cluster, eles podem avaliar o feedback separadamente.
# 
# Se uma tendência for encontrada em um cluster particular, ele permite que a empresa tome decisões direcionadas que beneficiam seus clientes, levando em conta o perfil deles. Isso é muito melhor do que generalizaria toda a base de clientes.

# ### Questão 11
# A estrutura adicional é derivada dos dados não rotulados originalmente quando utilizado as técnicas de clustering. Dado que cada cliente tem um **segmento de cliente** que melhor se identifica (dependendo do algoritmo de clustering aplicado), podemos considerar os *segmentos de cliente* como um **atributo construído (engineered)** para os dados. Assumindo que o distribuidor de atacado adquiriu recentemente dez novos clientes e cada um deles forneceu estimativas dos gastos anuais para cada categoria de produto. Sabendo dessas estimativas, o distribuidor de atacado quer classificar cada novo cliente em uma **segmentação de clientes** para determinar o serviço de entrega mais apropriado.  
# *Como o distribuidor de atacado pode rotular os novos clientes utilizando apenas a estimativa de despesas com produtos e os dados de* ***segmentação de clientes*** *  
# **Dica:** Um aprendiz supervisionado pode ser utilizado para treinar os clientes originais. Qual seria a variável alvo?

# **Resposta:**
# 
# É possível usar técnicas de aprendizado semi-supervisionado para classificar os clientes novos:
# 
# - Ao usar, inicialmente, uma abordagem de "clusterização", como o GMM, estabelecemos clusters e usamos isso como uma nova feature. Podemos chamar essa nova feature de "Segmento de clientes" e podemos atribuir valores para as classes (como 1 e 2) para estes novos clientes 
# 
# - Em seguida, podemos usar uma técnica de aprendizagem supervisionada, por exemplo, uma SVM  com uma variável alvo de "Segmento de Cliente"
# 
# - Podemos então otimizar o modelo de aprendizagem supervisionada para melhor classificar os novos clientes.

# ### Visualizando Distribuições Subjacentes
# 
# No começo deste projeto, foi discutido que os atributos `'Channel'` e `'Region'` seriam excluídos do conjunto de dados, então as categorias de produtos do cliente seriam enfatizadas na análise. Ao reintroduzir o atributo `'Channel'` ao conjunto de dados, uma estrutura interessante surge quando consideramos a mesma redução de dimensionalidade da PCA aplicada anteriormente no conjunto de dados original.
# 
# Execute o código abaixo para qual ponto de dados é rotulado como`'HoReCa'` (Hotel/Restaurante/Café) ou o espaço reduzido `'Retail'`. Al´´em disso, você vai encontrar as amostras de pontos circuladas no corpo, que identificará seu rótulo.

# In[19]:


# Mostre os resultados do clustering baseado nos dados do 'Channel'
rs.channel_results(reduced_data, outliers, pca_samples)


# ### Questão 12
# *Quão bom é o algoritmo de clustering e o números de clusters que você escolheu comparado a essa distribuição subjacente de clientes de Hotel/Restaurante/Café a um cliente Varejista? Há segmentos de clientes que podem ser classificados puramente como 'Varejistas' ou 'Hotéis/Restaurantes/Cafés' nessa distribuição? Você consideraria essas classificações como consistentes comparada a sua definição de segmentação de clientes anterior?*

# **Resposta:**
# 
# - Os dados reais se parecem muito com nossos clusters gerados anteriormente. Isso mostra que o agrupamento GMM conseguiu estabelecer as relações muito bem. Não foi possível capturar alguns dos pontos de dados, em especial os varejistas no cluster Hotel/Restaurante/Café.
# 
# - Os dados reais tem uma separação menos definida entre os clusters, mas podemos dizer que os datapoints com o 1º PC (menor que 4) e o 2º PC (menor que 2) são provavelmente um varejista. 
# 
# - Sim. São quase as suposições feitas em relação a classificação, Cluster 1 restaurantes/cafés e o Cluster 2 um Distribuidor ou supermercado, o que é equivalente aos varejistas.

# > **Nota**: Uma vez que você completou todas as implementações de código e respondeu todas as questões acima com êxito, você pode finalizar seu trabalho exportando um iPython Notebook como um documento HTML. Você pode fazer isso utilizando o menu acima e navegando até  
# **File -> Download as -> HTML (.html)**. Inclua o documento finalizado junto com esse Notebook para o seu envio.
