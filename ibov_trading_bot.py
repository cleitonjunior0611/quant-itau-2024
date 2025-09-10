import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs de TensorFlow

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suprimir logs de TensorFlow

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import matplotlib.pyplot as plt
import ta  # Biblioteca para indicadores técnicos


# 1. Instalação da Biblioteca 'ta'
# Caso ainda não tenha a biblioteca instalada, descomente a linha abaixo:
# !pip install ta

# 2. Baixar os dados diários do IBOVESPA de 2010 a 2020
def baixar_dados(ticker='^BVSP', start='2010-01-01', end='2020-01-01', interval='1d'):
    dados = yf.download(ticker, start=start, end=end, interval=interval)
    if dados.empty:
        raise ValueError("Nenhum dado foi baixado. Verifique o ticker e as configurações de data e intervalo.")
    return dados


# 3. Preprocessamento dos dados
def preprocessar_dados(dados, sequence_length=60):
    # Usar apenas a coluna 'Adj Close' (preço ajustado)
    data = dados['Adj Close'].values.reshape(-1, 1)

    # Normalizar os dados (entre 0 e 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Criar sequências de dados para o LSTM
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    # Converter para arrays numpy
    X, y = np.array(X), np.array(y)

    # Reshape para [amostras, passos de tempo, 1] (necessário para LSTM)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


# 4. Dividir os dados em treino e teste
def dividir_dados(X, y, train_size=0.5):
    train_len = int(len(X) * train_size)
    X_train, X_test = X[:train_len], X[train_len:]
    y_train, y_test = y[:train_len], y[train_len:]
    return X_train, X_test, y_train, y_test


# 5. Construir o modelo LSTM com os melhores hiperparâmetros
def construir_modelo():
    model = Sequential()

    # Camada de Entrada
    model.add(Input(shape=(60, 1)))  # sequence_length=60

    # Primeira camada LSTM com 128 unidades e dropout de 0.4
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(rate=0.4))

    # Segunda camada LSTM com 128 unidades e dropout de 0.1
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(rate=0.1))

    # Camada densa de saída
    model.add(Dense(units=1))

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# 6. Treinar o modelo
def treinar_modelo(model, X_train, y_train, epochs=34, batch_size=64):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return history


# 7. Fazer previsões
def fazer_previsoes(model, X_test, scaler):
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    return predicted_prices


# 8. Calcular Indicadores Técnicos (RSI)
def calcular_indicadores(dados, window=14):
    # Garantir que 'Adj Close' seja uma série 1D
    close_prices = dados['Adj Close'].squeeze()

    # Calcular o RSI
    rsi = ta.momentum.RSIIndicator(close=close_prices, window=window).rsi()

    # Adicionar o RSI ao DataFrame
    dados['RSI'] = rsi
    return dados


# 9. Gerar sinais de compra e venda com múltiplos indicadores
def gerar_sinais(test_df, rsi_buy=30, rsi_sell=70):
    buy_signals = []
    sell_signals = []
    position = None  # 'buy' ou 'sell'

    compras = []
    vendas = []

    for i in range(1, len(test_df)):
        # Condição de Compra: Previsão em alta e RSI < rsi_buy (sobrevendido)
        if (test_df.loc[i, 'Predicted_Price'] > test_df.loc[i - 1, 'Predicted_Price']) and (
                test_df.loc[i, 'RSI'] < rsi_buy):
            if position != 'buy':
                buy_signals.append(i)
                compras.append(test_df.loc[i, 'Real_Price'])
                position = 'buy'

        # Condição de Venda: Previsão em baixa e RSI > rsi_sell (sobrecomprado)
        elif (test_df.loc[i, 'Predicted_Price'] < test_df.loc[i - 1, 'Predicted_Price']) and (
                test_df.loc[i, 'RSI'] > rsi_sell):
            if position == 'buy':
                sell_signals.append(i)
                vendas.append(test_df.loc[i, 'Real_Price'])
                position = 'sell'

    return buy_signals, sell_signals, compras, vendas


# 10. Aplicar a Taxa de Venda e Simular o Capital
def simular_trading(compras, vendas, taxa_venda=0.001, capital_inicial=100.0):
    # Criar DataFrame de compras e vendas
    compras = pd.Series(compras)
    vendas = pd.Series(vendas)

    # Ajustar o tamanho de vendas para evitar desalinhamento
    if len(vendas) < len(compras):
        vendas = pd.concat([vendas, pd.Series([np.nan] * (len(compras) - len(vendas)))], ignore_index=True)

    compras_vendas_df = pd.DataFrame({
        'Comprado': compras,
        'Vendido': vendas
    })

    # Calcular o resultado de cada operação
    compras_vendas_df['Resultado'] = (compras_vendas_df['Vendido'] * (1 - taxa_venda) - compras_vendas_df['Comprado']) / \
                                     compras_vendas_df['Comprado']

    # Inicializar o capital
    capital_atual = capital_inicial
    capital_list = []

    # Simular as operações
    for index, row in compras_vendas_df.iterrows():
        comprado = row['Comprado']
        vendido = row['Vendido']

        if pd.notna(comprado) and pd.notna(vendido):
            capital_atual = capital_atual * (1 + row['Resultado'])
        capital_list.append(capital_atual)

    # Ajustar o tamanho do capital_list
    if len(capital_list) < len(compras_vendas_df):
        capital_list.extend([np.nan] * (len(compras_vendas_df) - len(capital_list)))

    compras_vendas_df['Capital'] = capital_list

    return compras_vendas_df, capital_atual


# 11. Visualizar os Resultados
def visualizar_resultados(test_df, buy_signals, sell_signals, compras_vendas_df):
    # Plotar os resultados
    plt.figure(figsize=(14, 7))

    # Preço Real
    plt.plot(test_df['Date'], test_df['Real_Price'], color='blue', label='Preço Real do IBOVESPA')

    # Previsões
    plt.plot(test_df['Date'], test_df['Predicted_Price'], color='red', label='Previsão LSTM do IBOVESPA')

    # Sinais de Compra
    plt.scatter(test_df.loc[buy_signals, 'Date'], test_df.loc[buy_signals, 'Predicted_Price'], marker='^',
                color='green', label='Sinal de Compra', s=100)

    # Sinais de Venda
    plt.scatter(test_df.loc[sell_signals, 'Date'], test_df.loc[sell_signals, 'Predicted_Price'], marker='v',
                color='red', label='Sinal de Venda', s=100)

    plt.title('Previsão de Preço do IBOVESPA com Sinais de Compra e Venda (LSTM)')
    plt.xlabel('Data')
    plt.ylabel('Preço (R$)')
    plt.legend()
    plt.show()

    # Plotar a evolução do capital
    plt.figure(figsize=(14, 7))
    plt.plot(compras_vendas_df['Capital'], marker='o', linestyle='-', color='purple', label='Capital')
    plt.title('Evolução do Capital Após as Operações de Trading')
    plt.xlabel('Operações')
    plt.ylabel('Capital (R$)')
    plt.legend()
    plt.show()


# 12. Função Principal para Executar Todo o Processo
def main():
    # Baixar os dados
    dados = baixar_dados()

    # Preprocessar os dados
    X, y, scaler = preprocessar_dados(dados)

    # Dividir os dados
    X_train, X_test, y_train, y_test = dividir_dados(X, y)

    # Construir o modelo com os melhores hiperparâmetros
    model = construir_modelo()

    # Treinar o modelo
    print("Treinando o modelo...")
    history = treinar_modelo(model, X_train, y_train)

    # Fazer previsões
    print("Fazendo previsões...")
    predicted_prices = fazer_previsoes(model, X_test, scaler)

    # Inverter a normalização dos dados de teste
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Criar DataFrame de teste com as previsões
    test_dates = dados.index[len(X_train) + 60:]  # sequence_length=60
    test_df = pd.DataFrame({
        'Date': test_dates,
        'Real_Price': y_test_original.flatten(),
        'Predicted_Price': predicted_prices.flatten()
    }).reset_index(drop=True)

    # Calcular indicadores técnicos
    print("Calculando indicadores técnicos (RSI)...")
    dados_com_indicadores = calcular_indicadores(dados)

    # Mapear os indicadores técnicos para o DataFrame de teste
    # Garantir que o mapeamento esteja correto
    test_df['RSI'] = dados_com_indicadores['RSI'].values[len(dados_com_indicadores) - len(test_df):]

    # Verificar se o mapeamento está correto
    print(test_df[['Date', 'Real_Price', 'Predicted_Price', 'RSI']].head(20))

    # Gerar sinais de compra e venda com múltiplos indicadores
    print("Gerando sinais de compra e venda...")
    buy_signals, sell_signals, compras, vendas = gerar_sinais(test_df, rsi_buy=40, rsi_sell=60)  # Limiares ajustados

    # Verificar quantos sinais foram gerados
    print(f"Sinais de Compra: {len(buy_signals)}")
    print(f"Sinais de Venda: {len(sell_signals)}")

    # Simular as operações de trading
    print("Simulando operações de trading...")
    compras_vendas_df, capital_final = simular_trading(compras, vendas)

    # Exibir os resultados
    print("\nDataFrame com Preços de Compra, Venda, Resultado de cada operação e Capital após cada operação:")
    print(compras_vendas_df)

    print(f"\nCapital final após todas as operações: R$ {capital_final:.2f}")

    # Visualizar os resultados
    visualizar_resultados(test_df, buy_signals, sell_signals, compras_vendas_df)

    # (Opcional) Plotar a perda de treinamento e validação
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda de Treinamento vs Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()


# Executar a função principal
if __name__ == "__main__":
    main()
