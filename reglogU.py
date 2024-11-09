#!usr/bin/env python

import numpy as np
import pandas as pd


##Extração dos dados##
doencas_pre = pd.read_csv('~/Documentos/dados/dadosRegLog.csv',sep=';', encoding='utf-8')

##Análise Inicial##

print(doencas_pre.head())
doencas_pre.shape

##OBJETIVO: Analisar se existe uma  tendência de óbito entre pessoas do sexo feminino e masculino##

from collections import Counter
Counter(doencas_pre.cs_sexo)
sossep=';'
doencas_pre['cs_sexo'].value_counts() #contabilizando o número de pessoas pelo sexo

##Valores Missind (NAN)##

print(doencas_pre.isnull().sum())
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
