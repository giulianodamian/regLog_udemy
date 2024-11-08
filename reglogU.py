#!usr/bin/env python3

import numpy as np
import pandas as pd


##Extração dos dados##
doencas_pre = pd.read_csv('~/Documentos/dados/dadosRegLog.csv',sep=';', encoding='utf-8')

##Análise Inicial##

print(doencas_pre.head())
doencas_pre.shape