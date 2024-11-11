#!/home/giuliano/anaconda3/envs/regLog_u/bin/python

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