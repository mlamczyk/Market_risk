# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ryzyko rynkowe
# ## Importy

# %%
#pip install numpy pandas matplotlib yfinance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# %% [markdown]
# ## Pobranie danych przez API

# %% [markdown]
# Źródło danych Yahoo Finance: https://finance.yahoo.com/chart/NVDA.
#
# Zbiór cen akcji NVIDIA Corporation (NVDA) od początku 2023 roku zawiera kolumny:
# - Adj Close: skorygowana cena zamknięcia.
# - Close: cena zamknięcia.
# - High: najwyższa cena w ciągu dnia.
# - Low: najniższa cena w ciągu dnia.
# - Open: cena otwarcia.
# - Volume: liczba akcji, która zmieniła właściciela tego dnia.

# %%
data = yf.download("NVDA", start="2023-01-01", end="2023-12-31")

# %%
data.to_csv("data/NVDA_data.csv")

# %% [markdown]
# ## Eksploracyjna analiza danych

# %%
data = pd.read_csv("data/NVDA_data.csv", header=[0, 1], index_col=0, parse_dates=True)
data.shape

# %%
data.columns

# %%
data.head()

# %% [markdown]
# Spłaszczenie oznaczeń kolumn:

# %%
data.columns = ["_".join(col).strip() for col in data.columns]
data.head()

# %%
data.tail()

# %% [markdown]
# Po wyświetleniu paru pierwszych i ostatnich wierszy widzimy, że brakuje obserwacji z niektórych dni (mamy 250 obserwacji w ciągu całego 2023 roku).

# %%
data.info()

# %% [markdown]
# Wykres szeregu czasowego cen zamknięcia:

# %%
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close_NVDA"], label="Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Close Price of NVDA Over Time")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ---

# %%
# Analiza danych – sprawdzenie zakresu i jakości danych (częstotliwość, wartości odstające, rozkład).
# Kalkulacja VaR – przy założeniu normalnego rozkładu (trzeba dopasować mu i sigma) oraz dla rozkładu historycznego z danych.
# Krótkie podsumowanie wyników – word, powerpoint lub dobrze skomentowany i opisany kod programistyczny.


# stopa zwrotu
# zysk lub starta

# dane do zyskow i strat, mamy powiedziec co chcemy liczyc, cos tam z var, test na rozklad normalny, komentarzy duzo co widzimy na wykresach

# %%
# Logarytmiczna stopa zwrotu jest preferowana przy analizach ryzyka, np. przy obliczaniu Value at Risk (VaR), ponieważ zapewnia addytywność.
# Przy dużych zyskach/stratach różnica między stopą prostą a logarytmiczną może być znacząca.
