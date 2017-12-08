
# coding: utf-8

# # Nanodegree Engenheiro de Machine Learning
# ## Aprendizagem Supervisionada
# ## Project 2: Construindo um Sistema de Intervenção para Estudantes

# Bem-vindo ao segundo projeto do Nanodegree de Machine Learning! Neste Notebook, alguns templates de código já foram fornecidos, e será o seu trabalho implementar funcionalidades necessárias para completar este projeto com êxito. Seções que começam com **'Implementação'** no cabeçalho indicam que o bloco de código que se segue precisará de funcionalidades adicionais que você deve fornecer. Instruções serão providenciadas para cada seção e as especificações para cada implementação estarão marcadas no bloco de código com o comando `'TODO'`. Tenha certeza de ler atentamente todas as instruções!
# 
# Além do código implementado, haverá questões relacionadas ao projeto e à implementação que você deve responder. Cada seção em que você tem que responder uma questão será antecedida de um cabeçalho **'Questão X'**. Leia atentamente cada questão e escreva respostas completas nas caixas de texto subsequentes que começam com **'Resposta: '**. O projeto enviado será avaliado baseado nas respostas para cada questão e a implementação que você forneceu.  
# 
# >**Nota:** Células de código e Markdown podem ser executadas utilizando o atalho de teclado **Shift + Enter**. Além disso, as células Markdown podem ser editadas, um clique duplo na célula entra no modo de edição.

# ### Questão 1 - Classificação versus Regressão
# *Seu objetivo neste projeto é identificar estudantes que possam precisar de intervenção antecipada antes de serem reprovados. Que tipo de problema de aprendizagem supervisionada é esse: classificação ou regressão? Por quê?*

# **Resposta: **
# 
# Eu acredito que para esse projeto classificação seria melhor. A razão é que queremos classificar os estudantes entre aqueles que precisarão de intervenção e aqueles que não o precisarão.

# ## Observando os Dados
# Execute a célula de código abaixo para carregar as bibliotecas de Python necessárias e os dados sobre os estudantes. Note que a última coluna desse conjunto de dados, `'passed'`, será nosso rótulo alvo (se o aluno foi ou não aprovado). As outras colunas são atributos sobre cada aluno.

# In[1]:


# Importar bibliotecas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from sklearn.metrics import f1_score

get_ipython().magic(u'matplotlib inline')

# Ler os dados dos estudantes
student_data = pd.read_csv("student-data.csv")
print "Os dados dos estudantes foram lidos com êxito!"


# ### Implementação: Observando os Dados
# Vamos começar observando o conjunto de dados para determinar quantos são os estudantes sobre os quais temos informações e entender a taxa de graduação entre esses estudantes. Na célula de código abaixo, você vai precisar calcular o seguinte:
# - O número total de estudantes, `n_students`.
# - O número total de atributos para cada estudante, `n_features`.
# - O número de estudantes aprovados, `n_passed`.
# - O número de estudantes reprovados, `n_failed`.
# - A taxa de graduação da classe, `grad_rate`, em porcentagem (%).
# 

# In[2]:


student_data.head()


# In[3]:


student_data.info()


# In[4]:


plt.figure(figsize = (16,6))
sns.set(style='whitegrid')

sns.heatmap(student_data.corr(), annot=True, cmap="YlGnBu", center=0);


# In[18]:


from __future__ import division

# TODO: Calculate number of students
n_students = student_data.shape[0]

# TODO: Calculate number of features
n_features = student_data[student_data.columns[:-1]].shape[1]

# TODO: Calculate passing students
n_passed = student_data[student_data.passed == 'yes'].shape[0]

# TODO: Calculate failing students
n_failed = student_data[student_data.passed == 'no'].shape[0]

# TODO: Calculate graduation rate
grad_rate = (n_passed / n_students) * 100

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## Preparando os Dados
# Nesta seção, vamos preparara os dados para modelagem, treinamento e teste.
# 
# ### Identificar atributos e variáveis-alvo
# É comum que os dados que você obteve contenham atributos não numéricos. Isso pode ser um problema, dado que a maioria dos algoritmos de machine learning esperam dados númericos para operar cálculos.
# 
# Execute a célula de código abaixo para separar os dados dos estudantes em atributos e variáveis-alvo e verificar se algum desses atributos é não numérico.

# In[19]:


# Extraia as colunas dos atributo
feature_cols = list(student_data.columns[:-1])

# Extraia a coluna-alvo, 'passed'
target_col = student_data.columns[-1] 

# Mostre a lista de colunas
print "Colunas de atributos:\n{}".format(feature_cols)
print "\nColuna-alvo: {}".format(target_col)

# Separe os dados em atributos e variáveis-alvo (X_all e y_all, respectivamente)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Mostre os atributos imprimindo as cinco primeiras linhas
print "\nFeature values:"
print X_all.head()


# ### Pré-processar Colunas de Atributo
# 
# Como você pode ver, há muitas colunas não numéricas que precisam ser convertidas! Muitas delas são simplesmente `yes`/`no`, por exemplo, a coluna `internet`. É razoável converter essas variáveis em valores (binários) `1`/`0`.
# 
# Outras colunas, como `Mjob` e `Fjob`, têm mais do que dois valores e são conhecidas como variáveis categóricas. A maneira recomendada de lidar com esse tipo de coluna é criar uma quantidade de colunas proporcional aos possíveis valores (por exemplo, `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc), e assinalar `1` para um deles e `0` para todos os outros.
# 
# Essas colunas geradas são por vezes chamadas de _variáveis postiças_ (em inglês: _dummy variables_), e nós iremos utilizar a função [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) para fazer essa conversão. Execute a célula de código abaixo para executar a rotina de pré-processamento discutida nesta seção.

# In[20]:


def preprocess_features(X):
    ''' Pré-processa os dados dos estudantes e converte as variáveis binárias não numéricas em
        variáveis binárias (0/1). Converte variáveis categóricas em variáveis postiças. '''
    
    # Inicialize nova saída DataFrame
    output = pd.DataFrame(index = X.index)

    # Observe os dados em cada coluna de atributos 
    for col, col_data in X.iteritems():
        
        # Se o tipo de dado for não numérico, substitua todos os valores yes/no por 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # Se o tipo de dado for categórico, converta-o para uma variável dummy
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Reúna as colunas revisadas
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))


# ### Implementação: Divisão dos Dados de Treinamento e Teste
# Até agora, nós convertemos todos os atributos _categóricos_ em valores numéricos. Para o próximo passo, vamos dividir os dados (tanto atributos como os rótulos correspondentes) em conjuntos de treinamento e teste. Na célula de código abaixo, você irá precisar implementar o seguinte:
# - Embaralhe aleatoriamente os dados (`X_all`, `y_all`) em subconjuntos de treinamento e teste.
#   - Utilizar 300 pontos de treinamento (aproxidamente 75%) e 95 pontos de teste (aproximadamente 25%).
#   - Estabelecer um `random_state` para as funções que você utiliza, se a opção existir.
#   - Armazene os resultados em `X_train`, `X_test`, `y_train` e `y_test`.

# In[24]:


# TODO: Import any additional functionality you may need here
from sklearn.cross_validation import train_test_split


# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    train_size=num_train, 
    random_state=42,
    stratify=y_all
)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ## Treinando e Avaliando Modelos
# Nesta seção, você irá escolher 3 modelos de aprendizagem supervisionada que sejam apropriados para esse problema e que estejam disponíveis no `scikit-learn`. Primeiro você irá discutir o raciocínio por trás da escolha desses três modelos considerando suas vantagens e desvantagens e o que você sabe sobre os dados. Depois você irá ajustar o modelo a diferentes tamanhos de conjuntos de treinamento (com 100, 200 e 300 pontos) e medir a pontuação F<sub>1</sub>. Você vai precisar preencher três tabelas (uma para cada modelo) que mostrem o tamanho do conjunto de treinamento, o tempo de treinamento, o tempo de previsão e a pontuação F<sub>1</sub> no conjunto de treinamento.
# 
# **Os seguintes modelos de aprendizagem supervisionada estão atualmente disponíveis no **[`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html)** para você escolher:**
# - Gaussian Naive Bayes (GaussianNB)
# - Árvores de Decisão
# - Métodos de agregação (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Método do gradiente estocástico (SGDC)
# - Máquinas de vetores de suporte (SVM)
# - Regressão logística

# ### Questão 2 - Aplicação dos Modelos
# *Liste três modelos de aprendizagem supervisionada que são apropriadas para esse problema. Para cada modelo escolhido:*
# - Descreva uma aplicação em mundo real na indústria em que o modelo pode ser aplicado. *(Talvez você precise fazer um pouco de pesquisa para responder essa questão – dê as devidas referências!)* 
# - Quais são as vantagens do modelo; quando ele tem desempenho melhor? 
# - Quais são as desvantagens do modelo, quando ele tem desempenho pior?
# - O que faz desse modelo um bom candidato para o problema, considerando o que você sabe sobre os dados?

# **Resposta: **
# 
# #### Support Vector Machine
# 
# O SVC, classificador que usa SVM, faz a classificação encontrando o hyperplane com distancia máxima entre as classes. SVM é usado com sucesso em datasets com muitas features em campos como bioinformática [1] e biologia [2]
# 
# As vantagens do SVM são sua performance e eficácia ao lidar com muitas variáveis, como no caso deste dataset. Outra vantagem do SVM é a possibilidade de usar o parametro kernel para separar as classes de forma não linear, o algoritmo faz isso adicionando novas dimensões usando valores calculados e assim encontrando o hyperplane que melhor separe as classes usando estas novas dimensões adicionadas.
# 
# As desvantagens podem ser a dificuldade em encontrar o conjunto de hyperparameters correto, como o kernel.
# 
# Eu escolhoi o SVM por ele ser um algoritmo muito usado, por performar bem quando adicionamos novas dimensões e ser um algoritmo que busca o hyperplane com a maior distancia entre as classes.
# 
# - [1] - https://www.ncbi.nlm.nih.gov/pubmed/15130823
# - [2] - https://noble.gs.washington.edu/papers/noble_support.html
# 
# 
# #### Logistic Regression
# 
# Regressão Logística é um modelo popular, que usa um threshold para produzir uma saída binária. Este modelo é usado em campos como testes A/B em marketing e industria financeira.
# 
# As vantagens são sua simplicidade e robustez que o torna menos propenso ao overfitting.
# 
# A principal desvantagem é que ele assume que as features podem ser linearmente separáveis
# 
# Ele foi escolhido por ter saída binária, pode ser eficiente e por ser pouco propenso ao overfitting.
# 
# 
# #### Random Forests
# 
# É a combinação de árvores de decisão que são criadas e treinadas individualmente. O algoritmo produz sua classificação ao calcular a moda das classificações de cada árvore de decisão individual. Um exemplo de aplicação é a previsão de preços de ações [1]
# 
# As vantagens são sua performance e eficiência em grandes volumes de dados tanto em relação aos exemplos de treinamento quanto de features.
# 
# As desvantagens são a possibilidade de overfitting, especialmente quando existem ruídos nos dados.
# 
# Ele foi escolhido por trabalhar bem com features binárias e por ter boa acurácia.
# 
# - [1] - http://www.scientific.net/AMM.740.947

# ### Configuração
# Execute a célula de código abaixo para inicializar três funções de ajuda que você pode utilizar para treinar e testar os três modelos de aprendizagem supervisionada que você escolheu acima. As funções são as seguintes:
# - `train_classifier` - recebe como parâmetro um classificador e dados de treinamento e ajusta o classificador aos dados.
# - `predict_labels` - recebe como parâmetro um classificador ajustado, atributos e rótulo alvo e faz estimativas utilizando a pontuação do F<sub>1</sub>.
# - `train_predict` - recebe como entrada um classificador, e dados de treinamento e teste, e executa `train_clasifier` e `predict_labels`.
#  - Essa função vai dar a pontuação F<sub>1</sub> tanto para os dados de treinamento como para os de teste, separadamente.

# In[23]:


def train_classifier(clf, X_train, y_train):
    ''' Ajusta um classificador para os dados de treinamento. '''
    
    # Inicia o relógio, treina o classificador e, então, para o relógio
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Imprime os resultados
    print "O modelo foi treinado em {:.4f} segundos".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''
    
    # Inicia o relógio, faz estimativas e, então, o relógio para
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Imprime os resultados de retorno
    print "As previsões foram feitas em {:.4f} segundos.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Treina e faz estimativas utilizando um classificador baseado na pontuação do F1. '''
    
    # Indica o tamanho do classificador e do conjunto de treinamento
    print "Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train))
    
    # Treina o classificador
    train_classifier(clf, X_train, y_train)
    
    # Imprime os resultados das estimativas de ambos treinamento e teste
    print "Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Implementação: Métricas de Desempenho do Modelo
# Com as funções acima, você vai importar os três modelos de aprendizagem supervisionada de sua escolha e executar a função `train_prediction` para cada um deles. Lembre-se de que você vai precisar treinar e usar cada classificador para três diferentes tamanhos de conjuntos de treinamentos: 100, 200 e 300 pontos. Então você deve ter 9 saídas diferentes abaixo – 3 para cada modelo utilizando cada tamanho de conjunto de treinamento. Na célula de código a seguir, você deve implementar o seguinte:
# - Importe os três modelos de aprendizagem supervisionada que você escolheu na seção anterior.
# - Inicialize os três modelos e armazene eles em `clf_A`, `clf_B` e `clf_C`.
#  - Defina um `random_state` para cada modelo, se a opção existir.
#  - **Nota:** Utilize as configurações padrão para cada modelo – você vai calibrar um modelo específico em uma seção posterior.
# - Crie diferentes tamanhos de conjuntos de treinamento para treinar cada modelo.
#  - *Não embaralhe e distribua novamente os dados! Os novos pontos de treinamento devem ser tirados de `X_train` e `y_train`.*
# - Treine cada modelo com cada tamanho de conjunto de treinamento e faça estimativas com o conjunto de teste (9 vezes no total).  
# **Nota:** Três tabelas são fornecidas depois da célula de código a seguir, nas quais você deve anotar seus resultados.

# In[30]:


# TODO: Import the three supervised learning models from sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the three models
clf_A = SVC(random_state=42)
clf_B = LogisticRegression(random_state=42)
clf_C = RandomForestClassifier(random_state=42)

# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]

X_train_300 = X_train
y_train_300 = y_train

# TODO: Execute the 'train_predict' function for each classifier and each training set size
# train_predict(clf, X_train, y_train, X_test, y_test)

for model in (clf_A, clf_B, clf_C):
    for size in (100, 200, 300):
        X = globals()['X_train_%s' % size]
        y = globals()['y_train_%s' % size]
        print
        train_predict(model, X, y, X_test, y_test)


# ### Resultados Tabulados
# Edite a célula abaixo e veja como a tabela pode ser desenhada em [Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#tables). Você deve salvar seus resultados abaixo nas tabelas fornecidas.

# ** Classifer 1 - SVC**  
# 
# | Training Set Size | Training Time           | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |        0.0023           |         0.0012         |      0.8625      |      0.8395     |
# | 200               |        0.0052           |         0.0019         |      0.8795      |      0.8312     |
# | 300               |        0.0089           |         0.0023         |      0.8747      |      0.8344     |
# 
# ** Classifer 2 - LogisticRegression**  
# 
# | Training Set Size | Training Time           | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |        0.0013           |         0.0002         |     0.9103       |     0.7324      |
# | 200               |        0.0020           |         0.0002         |     0.8502       |     0.7246      |
# | 300               |        0.0035           |         0.0002         |     0.8298       |     0.7482      |
# 
# ** Classifer 3 - RandomForestClassifier**  
# 
# | Training Set Size | Training Time           | Prediction Time (test) | F1 Score (train) | F1 Score (test) |
# | :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |
# | 100               |        0.0216           |         0.0010         |     1.0000       |     0.7519      |
# | 200               |        0.0226           |         0.0010         |     1.0000       |     0.7500      |
# | 300               |        0.0243           |         0.0010         |     0.9897       |     0.7188      |

# ## Escolhendo o Melhor Modelo
# Nesta seção final, você irá escolher dos três modelos de aprendizagem supervisionada o *melhor* para utilizar os dados dos estudantes. Você então executará um busca em matriz otimizada para o modelo em todo o conjunto de treinamento (`X_train` e `y_train`) ao calibrar pelo menos um parâmetro, melhorando em comparação a pontuação F<sub>1</sub> do modelo não calibrado. 

# ### Questão 3 - Escolhendo o Melhor Modelo
# *Baseando-se nos experimentos que você executou até agora, explique em um ou dois parágrafos ao conselho de supervisores qual modelo que você escolheu como o melhor. Qual modelo é o mais apropriado baseado nos dados disponíveis, recursos limitados, custo e desempenho?*

# **Resposta: **
# 
# Com base nos resultados obtidos, sem dúvida o melhor modelo é o SVM que obteve o melhor score no conjunto de testes, chegando a 83% de acurácia. Outro ponto a favor do SVM é que a diferença entre o score de treino e teste foi a menor, o que mostra menor tendencia ao overfit.

# ### Questão 4 – O Modelo para um Leigo
# *Em um ou dois parágrafos, explique para o conselho de supervisores, utilizando termos leigos, como o modelo final escolhido deve trabalhar. Tenha certeza que você esteja descrevendo as melhores qualidades do modelo, por exemplo, como o modelo é treinado e como ele faz uma estimativa. Evite jargões técnicos ou matemáticos, como descrever equações ou discutir a implementação do algoritmo.*

# **Resposta: **
# 
# O modelo de SVM está tentando encontrar algo chamado hiperplano - um limite de decisão que separa um exemplo de classe de outro, este é o caso dos alunos que passaram daqueles que não passaram. Este limite de decisão é ideal em termos da maior margem entre duas classes que estamos tentando separar. Então, quando sua tarefa é fazer uma previsão, o modelo usa esse limite para determinar qual classe atribuir ao novo data point - classe "passou" ou "não passou" com base na posição do novo data point em relação ao limite.
# 
# Quando a separação entre as classes não pode ser feita de forma linear, o SVM pode calcular novas features para produzir novas dimensões e, com as novas dimensões, encontrar um plano que separe as classes de forma linear. A este processo se da o nome de *"kernel trick"* e este processo pode se melhor entendido com a imagem abaixo:
# 
# ![rotate.gif](http://blog.pluskid.org/wp-content/uploads/2010/09/rotate.gif)
# 
# Nesta imagem vemos, no inicio, duas classes de dados que não podem ser separados de forma linear (ambas as classes fazem "uma curva"), mas ao decorrer do movimento vemos que em uma 3ª dimensão (profundidade) os dados podem ser separados de forma linear.

# ### Implementação: Calibrando o Modelo
# Calibre o modelo escolhido. Utilize busca em matriz (`GridSearchCV`) com, pelo menos, um parâmetro importante calibrado com, pelo menos, 3 valores diferentes. Você vai precisar utilizar todo o conjunto de treinamento para isso. Na célula de código abaixo, você deve implementar o seguinte:
# - Importe [`sklearn.grid_search.gridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html) e [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Crie o dicionário de parâmetros que você deseja calibrar para o modelo escolhido.
#  - Examplo: `parameters = {'parameter' : [list of values]}`.
# - Inicialize o classificador que você escolheu e armazene-o em `clf`.
# - Crie a função de pontuação F<sub>1</sub> utilizando `make_scorer` e armazene-o em `f1_scorer`.
#  - Estabeleça o parâmetro `pos_label` para o valor correto!
# - Execute uma busca em matriz no classificador `clf` utilizando o `f1_scorer` como método de pontuação e armazene-o em `grid_obj`.
# - Treine o objeto de busca em matriz com os dados de treinamento (`X_train`, `y_train`) e armazene-o em `grid_obj`.

# In[29]:


# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.svm import SVC


# TODO: Create the parameters list you wish to tune
parameters = [
    {'C': [0.5, 1, 1.5], 'kernel': ['linear']},
    {'C': [0.5, 1, 1.5], 'degree':[2, 3, 4], 'kernel': ['poly']},
    {'C': [0.5, 1, 1.5], 'gamma': [0.5, 0.1, 0.01], 'kernel': ['rbf']}
]

# TODO: Initialize the classifier
clf = SVC(random_state=42)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(score_func=f1_score, pos_label='yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Best parameters: %s" % grid_obj.best_params_ 
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


# ### Questão 5 - Pontuação F<sub>1</sub> Final
# *Qual é a pontuação F<sub>1</sub> do modelo final para treinamento e teste? Como ele se compara ao modelo que não foi calibrado?*

# **Resposta: **
# 
# A pontuação F1 do modelo final para o conjunto de treinamento ficou em 0.9781 e para o conjunto de testes em 0.8258. O escore F1 para o conjunto de treinamento no modelo tunado é superior ao do modelo não tunado, o que não se repete no conjunto de teste.

# > **Nota**: Uma vez que você completou todas as implementações de código e respondeu todas as questões acima com êxito, você pode finalizar seu trabalho exportando o iPython Nothebook como um document HTML. Você pode fazer isso utilizando o menu acima e navegando para  
# **File -> Download as -> HTML (.html)**. Inclua a documentação final junto com o notebook para o envio do seu projeto.
