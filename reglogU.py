#!/home/giuliano/anaconda3/envs/regLog_u/bin/python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd


##Extração dos dados##
doencas_pre = pd.read_csv('~/Documentos/dados/dadosRegLog.csv',sep=';', encoding='utf-8')

##Análise Inicial##

doencas_pre.head()
doencas_pre.shape

##OBJETIVO: Analisar se existe uma  tendência de óbito entre pessoas do sexo feminino e masculino##

from collections import Counter
Counter(doencas_pre.cs_sexo)
sossep=';'
doencas_pre['cs_sexo'].value_counts() #contabilizando o número de pessoas pelo sexo

##Valores Missind (NAN)##

doencas_pre.isnull().sum()
##Excluir valores NAN de cs_sexo##
doencas_pre.dropna(subset=['cs_sexo'], inplace=True)
##Excluir IGNORADO##
relacao = doencas_pre.loc[doencas_pre.cs_sexo != 'IGNORADO']
##Excluir INDEFINIDO##
relacao = relacao.loc[relacao.cs_sexo != 'INDEFINIDO']

relacao['cs_sexo'].value_counts()

import plotly.express as px
px.pie(relacao, names="cs_sexo")

##Análise dos óbitos##
relacao.obito.value_counts()
px.pie(relacao, names='obito')

##Renomeando registros da variável obito##

relacao['obito'] = relacao['obito'].replace({0:'nao', 1:'sim'})

print(relacao.head())

print(relacao.dtypes)
print(relacao.obito.value_counts())

##Transformando em variáveis categóricas##

relacao['cs_sexo'] = relacao['cs_sexo'].astype('category')
relacao['obito'] = relacao['obito'].astype('category')
##########################################################################
##########################################################################
############################# MODELO 1 ###################################
#################### Uma variável independente ###########################
print('Modelo 1\n')
print('Variável dependente binária (dicotômica).\n')
print('Categorias mutuamente exclusivas (uma pessoa não pode estar em duas situações).\n')
print('Independência das observações (sem medidas repetidas).\n')

import statsmodels.api as sm
import statsmodels.formula.api as smf

## Análise do modelo:

## Estatisicamente significativo: p<=0,05
## Estatisticamente não é significativo: p>0,05
## Análise da Ausência de outliers e pontos de alavancagem deve estar entre -3 e 3

modelo1 = smf.glm(formula='obito ~ cs_sexo', data=relacao, family= sm.families.Binomial()).fit()
print(modelo1.summary())
modelo1.params
modelo_prova = smf.glm(formula='cs_sexo ~obito', data=relacao, family= sm.families.Binomial()).fit()
print(modelo_prova.summary())

##Razão de chance com Intervalo de confiança de 95%

razao = np.exp(modelo1.params[1])
print(razao)

##  CONCLUSÃO: com intervalo de confiança de 95%, os homens tem 63,97%
# a menos de chance de spbreviver do que as mulheres.

coef = 1/razao
print(coef)

## Estatisticamente, com intervalo de confiança de 95%, a chance de 
# uma pessoa do sexo masculino ir a óbito é 1,56 vezes maior do que 
# a chance de uma pessoa do sexo feminino.


######################################################################
######################################################################

################################### MODELO 2 #########################
######################## Mais de uma variável independente ###########

## Diabetes e sexo

import statsmodels.api as sm
import statsmodels.formula.api as smf

relacao['diabetes'].value_counts()

import plotly.express as px
px.pie(relacao, names='diabetes')

relacao2= relacao.loc[relacao.diabetes != 'IGNORADO']

px.pie(relacao2, names='diabetes')

## Antes da exclusão de ignorados em diabetes ##
px.pie(relacao, names='obito')

## Depois da exclusão de ignorados em diabetes ##

px.pie(relacao2, names = 'obito')

relacao2.dtypes
relacao2['diabetes'] = relacao2['diabetes'].astype('category')


##=====>> Criação do modelo 2

# Ausência de Multicolinearidade entre as variáveis independentes.
#Análise do Modelo:
# Estatisticamente significativo p<=0,05
# Estatisticamente não é significativo p>0,05
# Análise da ausência de outliers e pontos de alavancagem
# Deve estar entre -3 e 3

modelo2 = smf.glm(formula='obito ~ cs_sexo + diabetes', data = relacao2, family = sm.families.Binomial()).fit()
print(modelo2.summary())
modelo2.params

## Chance com intervalo de confiança de 95% dos homens como relação às mulheres 

chance = 1/ (np.exp(modelo2.params[2]))
chance

## Conclusão: o resultado da diabetes está inconsistente devido a presença enorme de dados ignorados.

################################# Modelo 3 #############################################
########################################################################################

## Variável Idade

relacao3 = doencas_pre.loc[doencas_pre.nome_munic == 'Santos']
relacao3.head()
relacao3.shape
relacao3.dtypes

## Valores Missing (NAN)

relacao3.isnull().sum()

# Excluir valores missing
relacao3.dropna(subset=['idade'], inplace=True)

import matplotlib.pyplot as plt
plt.scatter(relacao3.idade,relacao3.obito)
plt.xlabel('IDADE')
plt.ylabel('ÓBITO')
plt.grid(False)
#plt.show()

#Ausência de multicolinearidade
np.corrcoef(relacao3.obito, relacao3.idade)

## Criação do modelo 3 com StatsModels

import statsmodels.api as sm
import statsmodels.formula.api as smf

modelo3 = smf.glm(formula='obito ~ idade', data=relacao3, family = sm.families.Binomial()).fit()
print(modelo3.summary())

'''summary = modelo3.summary().as_text()
with open('summary.txt','w') as f:
    f.write(summary)'''

#Razão de chance com intervalo de confiança de 95%

print(np.exp(modelo3.params[1]))

# Conclusão: Para cada ano mais velho, o indivíduo fica com 1,12 chances de outro indivíduo a menos.

###########################################################################

###++++++>>> Modelo 3 com Sklearn

from sklearn.linear_model import LogisticRegression
relacao3.head()
## Criação das variáveis x (independentes) e y (dependente)
# Transformação de X para o formato de matriz.

x = relacao3.iloc[:,2].values
y = relacao3.iloc[:,6].values

# Transformação de x para o formato de matriz

x = x.reshape(-1,1)


modelo3s = LogisticRegression()
modelo3s.fit(x,y)
modelo3s.coef_
modelo3s.intercept_

#Razão de chance com intervalo de confiança de 95%
np.exp(modelo3s.coef_)

# Para cada ano mais velho, o indivíduo fica com 1,12 das chances de outro indivíduo com um ano a menos.

plt.scatter(x,y)
# Geração de novos dados para gerar a função sigmoide
x_teste = np.linspace(0, 130, 100)

def model(w): # função sigmoide
    return 1 / (1+np.exp(-w))
# Geração de previsões (variável r) e visualização dos resultados
previsao = model(x_teste * modelo3s.coef_ + modelo3s.intercept_).ravel()
plt.plot(x_teste, previsao, color = 'red')
#plt.show()

# Testando o modelo com os resultados de outra cidade (Jundiaí)

jundiai = doencas_pre.loc[doencas_pre.nome_munic == 'Jundiaí']
jundiai.head()
jundiai.shape

#Valores Missin (NAN)

jundiai.isnull().sum()
# Excluir valores missing
jundiai.dropna(subset=['idade'], inplace=True)
#Mudança dos dados para o formato de matriz
idade = jundiai.iloc[:,2].values
idade = idade.reshape(-1,1)

# Previsões e geração da noca base de dados com valores originais  e as previsões
previsoes_teste = modelo3s.predict(idade)
print(previsoes_teste)

jundiai['previsões'] = previsoes_teste
jundiai = jundiai.drop(columns=['obito', 'previsões']).assign(obito=jundiai['obito'], previsoes=jundiai['previsões'])
jundiai.head(30)
jundiai['resultado'] = jundiai['obito'] + jundiai['previsoes']
jundiai.head(25)
jundiai["resultado"] = jundiai["resultado"].replace({0:"acertou", 1:"errou", 2:"acertou"})
jundiai.head(25)
jundiai['resultado'].value_counts()
px.pie(jundiai, names="resultado").show()
jundiai.to_csv('resultados_jundiai.csv', encoding = 'iso-8859-1', index = False)