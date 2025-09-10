# 📈 Predição do IBOVESPA com LSTM e Análise Técnica

Este projeto implementa um modelo de redes neurais recorrentes (LSTM) para prever os preços do **IBOVESPA** utilizando dados históricos do Yahoo Finance.  
Além da previsão, são aplicados **indicadores técnicos (RSI)** para gerar **sinais de compra e venda**, simulando operações de trading com cálculo de capital acumulado.

## ⚙️ Funcionalidades

- 📊 **Download automático** de dados históricos do IBOVESPA via `yfinance`
- 🔄 **Pré-processamento** dos dados com normalização e criação de janelas temporais
- 🧠 **Modelo LSTM otimizado** para previsão de séries temporais financeiras
- 📉 **Indicador RSI** para identificar condições de sobrecompra e sobrevenda
- 💰 **Simulação de operações de trading**, com taxa de venda e evolução do capital
- 📈 **Visualizações gráficas**:
  - Preço real x preço previsto
  - Sinais de compra e venda
  - Evolução do capital após cada operação
  - Curva de perda (treinamento vs validação)

## 🛠️ Tecnologias Utilizadas

- [Python 3.9+](https://www.python.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [ta (Technical Analysis)](https://technical-analysis-library-in-python.readthedocs.io/)
