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
#pip install numpy pandas matplotlib yfinance scipy pymannkendall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import pymannkendall as mk

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
print(data.shape)

# %%
print(data.columns)

# %%
print(data.head())

# %% [markdown]
# Spłaszczenie oznaczeń kolumn:

# %%
data.columns = ["_".join(col).strip() for col in data.columns]
print(data.head())

# %%
print(data.tail())

# %% [markdown]
# Po wyświetleniu paru pierwszych i ostatnich wierszy widzimy, że brakuje obserwacji z niektórych dni (mamy 250 obserwacji w ciągu całego 2023 roku).

# %%
print(data.info())

# %% [markdown]
# Wykres szeregu czasowego cen zamknięcia:

# %%
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close_NVDA"], color='blue', label="Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Close Price of NVDA Over Time")
plt.legend()
plt.grid()
plt.show()

# %%
dates = np.arange(len(data))  # Zastępujemy daty liczbami, aby ułatwić obliczenia
close_prices = data["Close_NVDA"]

# Obliczamy współczynniki linii trendu
slope, intercept = np.polyfit(dates, close_prices, 1)

# Obliczamy wartości linii trendu
trend_line = slope * dates + intercept

# Tworzymy wykres z linią trendu
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close_NVDA"], label="Close Price", color='blue')
plt.plot(data.index, trend_line, label="Trend Line", color='red', linestyle="--")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Close Price of NVDA Over Time with Trend Line")
plt.legend()
plt.grid()
plt.show()

# %%
trend_test = mk.original_test(data['Close_NVDA'])

# Wynik testu
print(trend_test.trend) # mamy trend rosnący

# %% [markdown]
# ## Prosta stopa zwrotu

# %%
# dzienne stopy zwrotu
data['daily_return'] = data['Close_NVDA'].pct_change() * 100 # w procentach

# %%
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["daily_return"], label="Close Price")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.title("Close Price of NVDA Over Time")
plt.legend()
plt.grid()
plt.show()

# %%
dates = np.arange(len(data))
daily_returns = data["daily_return"]

# Obliczamy współczynniki linii trendu
slope, intercept = np.polyfit(dates, daily_returns, 1)

# Obliczamy wartości linii trendu
trend_line = slope * dates + intercept

# Tworzymy wykres stóp zwrotu z linią trendu
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["daily_return"], label="Daily Return", color='blue')
plt.plot(data.index, trend_line, label="Trend Line", color='red', linestyle="--")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("Daily Return of NVDA Over Time with Trend Line")
plt.legend()
plt.grid()
plt.show()

# nie ma trendu

# %%
trend_test = mk.original_test(data['daily_return'])

# Wynik testu
print(trend_test.trend)

# %%
data.index.to_series().diff().value_counts()
# luki w datach to dni, w których rynki były zamknięte (np. weekendy, święta)
# różnice w datach nie przeszkadzają w policzeniu VaR

# %%
# wartości odstające
plt.figure(figsize=(8, 5))
data['daily_return'].plot(kind='box')
plt.title("Box Plot of Daily Returns")
plt.show()

# %%
# analiza rozkładu - histogram
plt.figure(figsize=(10, 6))
data['daily_return'].hist(bins=50, edgecolor='k', color='orange')
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Returns (%)")
plt.show()

# %%
# wnioski...
# gruby ogon z prawej strony
# Jeśli używasz dziennych stóp zwrotu, mogą one nie być rozkładem normalnym, ponieważ dane finansowe często mają "grube ogony"
# i lepiej mogą pasować do rozkładów typu t-Studenta.

# %% [markdown]
# - Wartości dodatnie oznaczają wzrost stopy zwrotu, czyli zysk.
# - Wartości ujemne to spadek stopy zwrotu, czyli strata.
# - Jeśli stopa zwrotu wynosi zero, oznacza to brak zmiany ceny.

# %% [markdown]
# ## Kalkulacja VaR
# ### a) Obliczanie VaR przy założeniu rozkładu normalnego

# %%
mu = data['daily_return'].mean()
sigma = data['daily_return'].std()

# %%
alpha_95 = 0.05
alpha_99 = 0.01

# %%
# VaR_95 = norm.ppf(1 - alpha_95, mu, sigma)
# VaR_99 = norm.ppf(1 - alpha_99, mu, sigma)
VaR_95_param = mu + sigma * np.percentile(np.random.normal(0, 1, 100000), 5)  # Parametryczny VaR dla 95%
VaR_99_param = mu + sigma * np.percentile(np.random.normal(0, 1, 100000), 1)  # Parametryczny VaR dla 99%

# %%
print(f"VaR 95% (Normalny rozkład): {VaR_95_param}")
print(f"VaR 99% (Normalny rozkład): {VaR_99_param}")

# %% [markdown]
# ### b) Obliczanie VaR dla rozkładu historycznego

# %%
# sortowanie stóp zwrotu
sorted_returns = data['daily_return'].sort_values()

# %%
VaR_hist_95 = sorted_returns.quantile(alpha_95)
VaR_hist_99 = sorted_returns.quantile(alpha_99)

# %%
print(f"VaR 95% (Rozkład historyczny): {VaR_hist_95}")
print(f"VaR 99% (Rozkład historyczny): {VaR_hist_99}")

# %%
plt.figure(figsize=(10, 6))
data['daily_return'].hist(bins=50, edgecolor='k', color='orange')
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.title("Histogram of Daily Returns (%)")

# Dodanie linii VaR dla podejścia historycznego
plt.axvline(VaR_hist_95, color='black', linestyle='--', linewidth=1, label='VaR 95% Historyczny')
plt.axvline(VaR_hist_99, color='black', linestyle='-', linewidth=1, label='VaR 99% Historyczny')

# Dodanie linii VaR dla podejścia parametrycznego
plt.axvline(VaR_95_param, color='red', linestyle='--', linewidth=1, label='VaR 95% Parametryczny')
plt.axvline(VaR_99_param, color='red', linestyle='-', linewidth=1, label='VaR 99% Parametryczny')

plt.legend()
plt.show()

# %%
# wnioski...

# %% [markdown]
# ## Podsumowanie wyników

# %%

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
