import pandas as pd

#import da tabela
tabela = pd.read_csv("advertising.csv")

#verifficar se tabela esta ok
#print(tabela.info())

import seaborn as sns
import matplotlib.pyplot as plt
#analise da tabela
#print(tabela.corr())
#sns.heatmap(tabela.corr(),cmap="PuBuGn",annot=True)
#plt.show()

#dividindo base p/ treino e teste
from sklearn.model_selection import train_test_split
y = tabela["Vendas"]
x = tabela[["TV","Radio", "Jornal"]] #ou= tabela.drop("Vendas", axis=1) quando for varias colunas
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y,test_size=0.3, random_state=1)

#inteligencia artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#treinar
modelo_regressaolinear.fit(x_treino,y_treino)
modelo_arvoredecisao.fit(x_treino,y_treino)
#teste
previsao_regressaolinear= modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao= modelo_arvoredecisao.predict(x_teste)

#comparar os dois modelos
from sklearn.metrics import r2_score

print(r2_score(y_teste,previsao_regressaolinear))
print(r2_score(y_teste,previsao_arvoredecisao)) #mais proximo

#fazendo previsao mes que vem
novomes= pd.read_csv("novos.csv")
print(novomes)
print(modelo_arvoredecisao.predict(novomes))