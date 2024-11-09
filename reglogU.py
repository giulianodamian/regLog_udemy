#!usr/bin/env python3

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