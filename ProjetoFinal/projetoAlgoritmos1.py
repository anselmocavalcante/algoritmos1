# Importação das bibliotecas

from typing import List
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from fpdf import FPDF
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Faz a leitura da base de dados
def lerArquivo ():
    # Arquivos disponível em:
    # glass.data (url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")

    # nome do arquivo da base da dados
    nomeArquivo = "glass.data"

    # nome das colunas da base de dados.
    nomesColunas = ['id_number', 'ri_refractive_index', 'na_sodium', 'mg_magnesium', 'al_aluminium', 'si_silicon','k_potassium','ca_calcium', 'ba_barium', 'fe_iron', 'type_of_glass']

    # Faz a leitura da base de dados e atribui a variável baseDeDados (dataframe)
    baseDeDados = pandas.read_csv(nomeArquivo, names=nomesColunas)

    # Imprime a base de dados na tela
    #print (baseDeDados)

    # nome da classe presente no arquivo de base de dados
    nomeDaClasse = "type_of_glass"

    # Seleciona apenas o conteudo da base refrente aos atributos
    tabelaDeAtributos = baseDeDados.drop([nomeDaClasse], axis=1)

    # Remove a coluna id_number da tabela de atributos
    tabelaDeAtributos = tabelaDeAtributos.drop(['id_number'], axis=1)

    # Seleciona apenas o conteudo da base refrente a classe
    classificacao = baseDeDados[nomeDaClasse]

    return tabelaDeAtributos, classificacao, nomeArquivo, nomeDaClasse

# Realiza a normalização dos dados
def normalizar (atributos:List) -> List:

    scaler = StandardScaler()
    dadosNormalizados = scaler.fit_transform(atributos)

    return dadosNormalizados

# solicita ao usuário quais modelos ele deseja aplicar a base de dados e retorna uma lista com o nome dos modelos
def lerModelos() -> List[str]:

    modelosDisponiveis: List[str] = ['mlp-linear', 'mlp-logistic', 'mlp-tanh', 'mlp-relu', 'svm-linear', 'svm-poly', 'svm-rbf','svm-sigmoid', 'random forest']

    modelosEscolhidos: List[str] = []
    opcao: str = "100"

    while opcao != "0":
        print("Escolha os modelos da lista para serem aplicados: ")
        print("Modelos disponíveis:")
        posicao:int = 0
        for m in modelosDisponiveis:
            posicao = posicao + 1
            print(posicao, "-", m)
        print ("Informe 0 para continuar e TODOS para escolher todos os modelos")
        print("Modelos escolhidos até o momento: ", modelosEscolhidos)
        opcao = input("informe o nome do modelo escolhido: ")

        if opcao == "0":
            return modelosEscolhidos

        if opcao.lower() == "todos":
            return ['mlp-linear', 'mlp-logistic', 'mlp-tanh', 'mlp-relu', 'svm-linear', 'svm-poly', 'svm-rbf','svm-sigmoid', 'random forest']

        if not checaElementoLista(modelosDisponiveis, opcao):
            input("A opção informada não está na lista")
        else:
            modelosEscolhidos.append(opcao.lower())
            modelosDisponiveis.pop(posicaoElemento(modelosDisponiveis,opcao.lower()))

        if len(modelosDisponiveis) == 0:
            input("Não há mais modelos disponíeis para seleção")
            return modelosEscolhidos

    return modelosEscolhidos


# Funcao que recebe uma lista de numeros e retorna o maior elemento da lista
def maiorElemento(numeros:List[float]) -> float:
    if len(numeros) == 0:
        raise RuntimeError("lista vazia")

    maior = numeros[0]
    for n in numeros:
        if n > maior:
            maior = n
    return maior

# Funcao que recebe uma lista de numeros e retorna o menor elemento da lista
def menorElemento(numeros:List[float]) -> float:
    if len(numeros) == 0:
        raise RuntimeError("lista vazia")

    menor = numeros[0]
    for n in numeros:
        if n < menor:
            menor = n
    return menor

# Função  que recebe uma lista e um elemento e checa o elemento pertence a lista
def checaElementoLista (lista, elemento) -> bool:

    for e in lista:
        if elemento.lower() == e.lower():
            return True
    return False

# Função que rece uma lista e um elemento e retorna a posicao desse elemento na lista
def posicaoElemento(lista:List, elemento) -> int:
    if len(lista) == 0:
        raise RuntimeError("lista vazia")

    posicao = 0
    for n in lista:
        if n == elemento:
            return posicao
        else:
            posicao = posicao + 1
    return -1

# Função que cria treina e testa o classificador e retorna as métricas e o classificador treinado
def criaTreinaTestaClassificador (atributosTreinameto:List, atributosTeste:List, classificacaoTreinamento:List, classificacaoTeste:List, modelo:str):

    if modelo != 'mlp-linear' and modelo != 'mlp-logistic' and modelo != 'mlp-tanh' and modelo != 'mlp-relu' and modelo != 'svm-linear' and modelo != 'svm-poly' and modelo != 'svm-rbf' and modelo != 'svm-sigmoid' and modelo != 'random forest':
        return

    classificador = svm.SVC(kernel='linear')

    if modelo == "mlp-linear":
        classificador = MLPClassifier(activation='identity', random_state=1, max_iter=1000, solver='sgd')
    if modelo == "mlp-logistic":
        classificador = MLPClassifier(activation='logistic', random_state=1, max_iter=1000, solver='sgd')
    if modelo == "mlp-tanh":
        classificador = MLPClassifier(activation='tanh', random_state=1, max_iter=1000, solver='sgd')
    if modelo == "mlp-tanh":
        classificador = MLPClassifier(activation='relu', random_state=1, max_iter=1000, solver='sgd')
    if modelo == "svm-poly":
        classificador = svm.SVC(kernel='poly')
    if modelo == "svm-rbf":
        classificador = svm.SVC(kernel='rbf')
    if modelo == "svm-sgmoid":
        classificador = svm.SVC(kernel='sigmoid')
    if modelo == "random forest":
        classificador = RandomForestClassifier(n_estimators=100)

    # treina o classificador
    classificador.fit(atributosTreinameto, classificacaoTreinamento)

    # realiza a predicao do classificador
    predicao = classificador.predict(atributosTeste)

    # compara a predicao com o valor correto e obtem a métrica acurácia
    acuracia = metrics.accuracy_score(classificacaoTeste, predicao)

    # compara a predicao com o valor correto e obtem a métrica f1-score
    f1Score = metrics.f1_score(classificacaoTeste, predicao, average='macro')

    # compara a predicao com o valor correto e obtem a métrica recall
    recall = metrics.recall_score(classificacaoTeste, predicao, average='macro')

    return acuracia, f1Score, recall, classificador

# função que recebe o classificador treinado e uma lista (amostra), realiza a predicão e retorna o valor da predicao
def fazPredicao (classificadorTreinado, amostra:List) -> int:

    # necessário para o método predict
    listaAmostras = []
    listaAmostras.append(amostra)
    return classificadorTreinado.predict(listaAmostras)[0]

# Função que rece uma string e verifica se a string corresponde a um numero
def ehNumero(string:str) -> bool:

    if string.isnumeric():
        return True
    else:
        try:
            float(string)
        except ValueError:
            return False
    return True

# função que solicita ao usuário se ele deseja realizar a predicao e em caso positivo lê os valores da amostra
def solictaPredicao(atributosTeste) -> List:

    opcao: str = "100"
    predicao: List = []

    while opcao != "0" and opcao != "1":
        print("Deseja informar os dados de um elemento para realizar predição?")
        print("Informe 0 para NÃO e 1 para SIM:")
        opcao = input("informe sua opção: ")

        if opcao != "0" and opcao != "1":
            input("A opção informada não é válida")

    if opcao == "0":
        return []

    if opcao == "1":
        listaDeAtributos:List = atributosTeste.columns.tolist()
        listaExemploValores = atributosTeste.values.tolist()[0]
        tamanhoLista:int = len(listaDeAtributos)
        contador:int = 0

        while contador != tamanhoLista:

            print("# Atributo",contador+1,"de",tamanhoLista,"#")
            print("Informe o valor para o atributo",listaDeAtributos[contador])
            print("Exemplo de valor:",listaExemploValores[contador])
            valor = input()
            if ehNumero(valor):
                predicao.append(float(valor))
                contador = contador + 1
            else:
                print("O valor informado não corresponde ao mesmo tipo do valor do exemplo")

    return predicao

# função principal
def projetoAlgoritmos1():

    #dados:List = lerArquivo()
    dados = lerArquivo()
    atributos:List = dados[0]
    classificacao:List = dados[1]
    nomeDaBase:str = dados[2]
    nomeDaClasse:str = dados[3]

    # separa os conjuntos de treinamento e teste
    atributosTreinameto, atributosTeste, classificacaoTreinamento, classificacaoTeste = train_test_split(atributos, classificacao, test_size=0.3, random_state=5)

    # Normaliza o conjunto de atributos utilizados para o treinamento e para o teste
    atributosTreinameto:List = normalizar(atributosTreinameto)
    atributosTeste:List = normalizar(atributosTeste)

    #Solicita ao usuario os modelos de treinamento
    modelos:List[str] = lerModelos()

    if modelos ==[]:
        print("Nenhum modelo foi selecionado.")
        print("Encerrando o programa...")
        return 0

    acuracias:List = []
    f1Score:List = []
    recall:List = []
    listaCLassificadoresTreinados:List = []

    # cria, treina  e obtem a acuracia dos classificadores
    for m in modelos:
        resultadoClassificador = criaTreinaTestaClassificador(atributosTreinameto, atributosTeste, classificacaoTreinamento, classificacaoTeste, m)
        acuracias.append(resultadoClassificador[0])
        f1Score.append(resultadoClassificador[1])
        recall.append(resultadoClassificador[2])
        listaCLassificadoresTreinados.append(resultadoClassificador[3])

    # imprime as na tela as listas de métricas
    #print(acuracias)
    #print(f1Score)
    #print(recall)

    # normaliza as listas das métricas para impressão na imagem
    for i in range (0, len(acuracias)):
        acuracias[i] = acuracias[i] * 100
        f1Score[i] = f1Score[i] * 100
        recall[i] = recall[i] * 100


    # Solicita a predição

    resultadoSolicitacaoPredicao:List = solictaPredicao(atributos)
    resultadoPredicao:List = []
    if resultadoSolicitacaoPredicao != []:
        contador: int = 0
        #print("# Predição #")
        for c in listaCLassificadoresTreinados:
            resultadoPredicao.append(fazPredicao(c,resultadoSolicitacaoPredicao))
            #print("Classificador", modelos[contador])
            #print("Classificação: ",resultadoPredicao[contador])
            contador = contador + 1


    # Plotagem dos gráficos

    #Formata nome dos modelos para plotagem
    for i in range (0, len(modelos)):
        if modelos[i] == "mlp-linear":
            modelos[i] = "MLP-li"
        if modelos[i] == "mlp-logistic":
            modelos[i] = "MLP-lo"
        if modelos[i] == "mlp-tanh":
            modelos[i] = "MLP-ta"
        if modelos[i] == "mlp-relu":
            modelos[i] = "MLP-re"
        if modelos[i] == "svm-linear":
            modelos[i] = "SVM-li"
        if modelos[i] == "svm-poly":
            modelos[i] = "SVM-po"
        if modelos[i] == "svm-rbf":
            modelos[i] = "SVM-rb"
        if modelos[i] == "svm-sigmoid":
            modelos[i] = "SVM-si"
        if modelos[i] == "random forest":
            modelos[i] = "RF"

    for i in range(1,4):
        if i == 1:
            metrica = acuracias
            nome = "acurácia"
        elif i == 2:
                metrica = f1Score
                nome = "f1-score"
        else:
            metrica = recall
            nome = "recall"

        plt.figure(figsize=(2.7, 1.4))
        plt.bar(modelos, metrica, width=0.5)
        plt.title('Figura %i' % i, fontsize=4)
        plt.xlabel('Modelos', fontsize=4)
        plt.ylabel('%s' % nome, fontsize=4)
        plt.tick_params(labelsize=4)

        axes = plt.gca()
        axes.set_ylim([0, 100])

        fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
        yticks = mtick.FormatStrFormatter(fmt)
        axes.yaxis.set_major_formatter(yticks)

        plt.savefig("%s" % nome, bbox_inches='tight', dpi=200)

        plt.show()


# Criacao do relatório

    # Cria o objeto pdf com uma pagina
    pdf = FPDF()
    pdf.add_page()

    # imprime a data e a hora de geracao do relatorio no pdf
    pdf.set_xy(0, 2)
    pdf.set_font('arial', '', 10)
    agora = datetime.now()
    dt_string = agora.strftime("%d/%m/%Y %H:%M:%S")
    pdf.cell(0, 5, dt_string, 0, 0, 'R')

    # imprime o titulo do relatorio no pdf
    pdf.set_xy(0, 10)
    pdf.cell(60)
    pdf.set_font('arial', 'B', 14)
    pdf.cell(90, 10, "Relatório sobre o desempenho dos modelos selecionaodos", border=0, ln=2, align='C')

    # imprime os dados da base no pdf
    pdf.set_xy(5, 25)
    pdf.set_font('arial', '', 10)
    pdf.cell(13)
    pdf.cell(90, 5, "Nome ou endereço da base: %s" % nomeDaBase, border=0, ln=1, align='L')
    pdf.cell(8)
    pdf.cell(90, 5, "Nome da classe: %s" % nomeDaClasse, border=0, ln=1, align='L')
    #pdf.cell(-6)

    if resultadoSolicitacaoPredicao != []:
        pdf.cell(8)
        pdf.cell(90, 5, "Elemento utilizado na predição: %s" % resultadoSolicitacaoPredicao, border=0, ln=1, align='L')

    # imprime a tabela do relatorio no pdf
    pdf.set_xy(55, 45)
    pdf.cell(90, 5, "Tabela 1 - Relação modelo x acurácia x f1-score x recall", border=0, ln=1, align='C')
    pdf.set_xy(55, 50)
    pdf.set_font('arial', 'B', 9)
    pdf.cell(20, 8, 'Modelo', 1, 0, 'C')
    pdf.cell(25, 8, 'Acurácia (%)', 1, 0, 'C')
    pdf.cell(25, 8, 'F1-score (%)', 1, 0, 'C')

    if resultadoSolicitacaoPredicao != []:
        pdf.cell(20, 8, 'Recall (%)', 1, 0, 'C')
        pdf.cell(20, 8, 'Predição', 1, 2, 'C')
        pdf.cell(-90)
    else:
        pdf.cell(25, 8, 'Recall (%)', 1, 2, 'C')
        pdf.cell(-70)
    pdf.set_font('arial', '', 8)

    for i in range(0, len(modelos)):
        pdf.cell(20, 8, '%s' % (modelos[i]), 1, 0, 'C')
        if acuracias[i] == menorElemento(acuracias) and len(modelos) > 1: # se menor elemento seta a cor vermelha
            pdf.set_fill_color(255, 0, 0)
        elif acuracias[i] == maiorElemento(acuracias) and len(modelos) > 1: # se maior elemento seta a cor verde
            pdf.set_fill_color(50, 205, 50)
        else: # se não é maior nem menor, seta a cor branca
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(25, 8, '%f' % (acuracias[i]), 1, 0, 'C', fill=True)

        if f1Score[i] == menorElemento(f1Score) and len(modelos) > 1: # se menor elemento seta a cor vermelha
            pdf.set_fill_color(255, 0, 0)
        elif f1Score[i] == maiorElemento(f1Score) and len(modelos) > 1: # se maior elemento seta a cor verde
            pdf.set_fill_color(50, 205, 50)
        else: # se não é maior nem menor, seta a cor branca
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(25, 8, '%f' % (f1Score[i]), 1, 0, 'C', fill=True)
        if resultadoSolicitacaoPredicao != []:
            if recall[i] == menorElemento(recall) and len(modelos) > 1:  # se menor elemento seta a cor vermelha
                pdf.set_fill_color(255, 0, 0)
            elif recall[i] == maiorElemento(recall) and len(modelos) > 1:  # se maior elemento seta a cor verde
                pdf.set_fill_color(50, 205, 50)
            else: # se não é maior nem menor, seta a cor branca
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(20, 8, '%f' % (recall[i]), 1, 0, 'C', fill=True)
            pdf.cell(20, 8, '%i' % (resultadoPredicao[i]), 1, 2, 'C')
            pdf.cell(-90)
        else:
            if recall[i] == menorElemento(recall) and len(modelos) > 1:  # se menor elemento seta a cor vermelha
                pdf.set_fill_color(255, 0, 0)
            elif recall[i] == maiorElemento(recall) and len(modelos) > 1:  # se maior elemento seta a cor verde
                pdf.set_fill_color(50, 205, 50)
            else: # se não é maior nem menor, seta a cor branca
                pdf.set_fill_color(255, 255, 255)
            pdf.cell(25, 8, '%f' % (recall[i]), 1, 2, 'C', fill=True)
            pdf.cell(-70)

    # carrega a imagem do grafico e imprime no relatorio pdf
    pdf.set_xy(40, 130)
    pdf.cell(90, 10, " ", 0, 2, 'C')
    pdf.cell(-30)
    pdf.image('acurácia.png', x = 5, y = None, w = 0, h = 0, type = '', link = '')
    pdf.image('f1-score.png', x=5, y=None, w=0, h=0, type='', link='')
    pdf.image('recall.png', x=5, y=None, w=0, h=0, type='', link='')

    # salva no disco o relatorio em formato pdf
    try:
        pdf.output('relatorio.pdf', 'F')
    except PermissionError:
        print("O relatório PDF não pôde ser salvo por falta de permissão")


# chama a funcao principal
projetoAlgoritmos1()
