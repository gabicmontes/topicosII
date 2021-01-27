import pandas as pd

base = pd.read_csv("hotel_booking_data.csv")

print("1. Quantas linhas a sua base de dados tem?\n")
linhas, colunas = base.shape
print("A base tem ", linhas, " linhas\n\n")

print("2. Existem campos com informação faltando na base? Se sim, informe a coluna com mais elementos faltantes.\n")
coluna = []
coluna_maior = []
maior = 0

for i in range (0, colunas-1):
    coluna = base.iloc[:, i:i+1]
    if(coluna.isna().sum()[0]  > maior):
        maior = coluna.isna().sum()[0] 
        coluna_maior = coluna

if(maior>0):
    print("Sim")
    print("A coluna com mais elementos faltantes é a ", coluna_maior.columns.tolist(), " contendo ", maior, " elementos nulos\n\n")
    
print("3.  Remova a coluna company da base de dados.\n")
base = base.drop('company', axis = 1)
print("Coluna company apagada com sucesso\n\n")

print("4. Quais são os 5 valores mais comuns da coluna country? Retorne-os, informando a ocorrência de cada um deles.\n")
print("País - Ocorrência")
country = base['country'].value_counts().head(5)
print(country,"\n\n")

print("5. Qual o nome da pessoa que mais paga por dia em média (average daily rate ou adr) na base de dados?\n")
id_nome = base['adr'].idxmax()
valor = base['adr'].max()
nome = base['name'][id_nome]

print("A pessoa que mais paga por dia em média é o ", nome, " pagando o valor de ", valor, "\n\n")

print("6. Informe o gasto diário médio (adr) dos hóspedes da base de dados.\n")
gasto_medio = base['adr'].mean()
print("O gasto diário médio é ", gasto_medio, "\n\n")

print("7. Diga o nome e e-mail das pessoas que fizeram ao menos 5 requisições especiais.\n")
requisicoes_especiais = base[(base["total_of_special_requests"] >= 5)]

print("Pessoas que fizeram ao menos 5 requisições especiais:")
print(requisicoes_especiais[['name', 'email']],"\n\n")

print("8. Retorne os 5 sobrenomes mais comuns das pessoas da base de dados.\n")
print("Sobrenome - Ocorrência")
sobrenome = base['name'].str.partition(' ')[[0, 2]]
sobrenome = sobrenome[2]
sobrenome_comum = sobrenome.value_counts().head(5)
print(sobrenome_comum)

