API - EXEMPLOS DE UTILIZAÇÃO
==============================

Este documento tem por objetivo documentar o código da API implementada
e como deve ser realizada a sua utilização.

VIT - Algoritmo de Classificador Binário
----------------------------------------

Esse módulo possui duas possíveis chamadas:
`localhost:5000/train_vit` e `localhost:5000/predict_vit`.

*1 - Método GET:*

A requisição abaixo efetua o treinamento do modelo

**Requisição:** localhost:5000/train_vit

**Exemplo de Saída: (JSON)**

.. code-block:: JSON

    {
        "1.356205042328445964e+00",
        "-1.509168416533169577e+00"
    }

*2 - Método POST:*

A requisição abaixo efetua a predição do modelo

**Requisição:** localhost:5000/predict_vit

**Exemplo de Entrada: (JSON)**

.. code-block:: JSON

    {
        "Exemplo"
    }

**Exemplo de Saída: (JSON)**

.. code-block:: JSON

    {
        "1.356205042328445964e+00",
        "-1.509168416533169577e+00"
    }
