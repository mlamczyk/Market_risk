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
#pip install numpy pandas seaborn matplotlib yfinance scipy pymannkendall statsmodels
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import pymannkendall as mk
from statsmodels.tsa.stattools import adfuller

# %% [markdown]
# ## Pobranie danych przez API

# %% [markdown]
# Źródło danych Yahoo Finance: https://finance.yahoo.com/chart/NVDA.

# %%
data = yf.download("NVDA", start="2023-01-01", end="2023-12-31")

# %%
#data.to_csv("data/NVDA_data.csv")

# %% [markdown]
# ## Eksploracyjna analiza danych

# %%
#data = pd.read_csv("data/NVDA_data.csv", header=[0, 1], index_col=0, parse_dates=True)
print(data.head())

# %% [markdown]
# Pobraliśmy dzienne dane cen akcji NVDA z 2023 roku. Możemy sprawdzić wymiary zbioru danych:

# %%
print(data.shape)

# %% [markdown]
# Zbiór ma sześć kolumn, ale tylko 250 wierszy (dni). Wynika to z tego, że dane są zapisywane tylko dla dni, w których rynki były otwarte, co oznacza, że weekendy i święta mogą być pominięte. Zatem liczba dni handlowych w 2023 roku wynosiła 250. Sprawdzimy różnice w indeksach:

# %%
data.index.to_series().diff().value_counts()

# %% [markdown]
# - Największa liczba dni handlowych (196) występuje z różnicą 1 dnia - większość kolejnych wpisów pochodzi z dni roboczych.
# - Różnica 3 dni odpowiada przerwom weekendowym, gdy giełda jest zamknięta w soboty i niedziele.
# - Różnica 4 dni wynika z przedłużonych weekendów.
# - Różnica 2 dni jest rzadsza i może wskazywać na skrócone tygodnie handlowe, gdy święto przypada na inny dzień niż piątek lub poniedziałek.
#
# Dane mają sześć kolumn o nazwach:

# %%
print(data.columns)

# %% [markdown]
# Nagłówek kolumn jest wielopoziomowy. Możemy go spłaszczyć, aby łatwiej odnosić się do konkretnych kolumn:

# %%
data.columns = ["_".join(col).strip() for col in data.columns]
print(data.columns)

# %% [markdown]
# Wyświetlimy podstawowe informacje o zbiorze danych:

# %%
print(data.info())

# %% [markdown]
# Indeks:
# - Indeks składa się z dat i godzin. Dane obejmują okres od 3 stycznia 2023 (pierwszy dzień handlowy) do 29 grudnia 2023 (ostatni dzień handlowy). W zbiorze jest 250 wierszy, co odpowiada liczbie dni handlowych w 2023 roku.
#
# Opis kolumn:
# - **Adj Close_NVDA**: skorygowana cena zamknięcia, uwzględnia takie zdarzenia jak podział akcji (split) czy dywidendy.
# - **Close_NVDA**: cena zamknięcia w danym dniu, czyli ostatnią cenę, po której akcje były handlowane na koniec dnia giełdowego.
# - **High_NVDA**: najwyższa cena w ciągu dnia handlowego.
# - **Low_NVDA**: najniższa cena w ciągu dnia handlowego.
# - **Open_NVDA**: cena otwarcia, czyli pierwsza cena, po której akcje były handlowane w danym dniu.
# - **Volume_NVDA**: liczba akcji, która zmieniła właściciela tego dnia.
#
# Typy danych w kolumnach:
# - float64 (5 kolumn): Kolumny z wartościami liczbowymi zmiennoprzecinkowymi (ceny akcji).
# - int64 (1 kolumna): Kolumna z wartościami całkowitymi (wolumen obrotu).
#
# Możemy też sprawdzić podstawowe statysktyki zbioru:

# %%
print(data.describe())

# %% [markdown]
# Wnioski:
# - Wszystkie kolumny mają 250 wartości (zgodne z liczbą dni handlowych w 2023 roku). Brak danych nie występuje.
# - Akcje miały średnią cenę na poziomie 36.56 USD, ale mediana 41.88 USD (percentyl 50%) wskazuje, że ceny częściej oscylowały wokół wyższych wartości.
# - Rozpiętość między minimum (14.26 USD) a maksimum (50.39 USD) jest bardzo duża, co sugeruje znaczne wahania cen w ciągu roku.
# - Rozkład cen w kwartylach pokazuje, że przez większość roku ceny akcji znajdowały się w przedziale ~27-46 USD.
# - Średni wolumen obrotu był wysoki (~473 milionów akcji dziennie), ale znaczne odchylenie standardowe (161,402,800 akcji) wskazuje na duże różnice w aktywności inwestorów w poszczególnych dniach.

# %% [markdown]
# Interesuje nas kolumna Close_NVDA. Możemy przedstawić szereg czasowy cen zamknięcia na wykresie:

# %%
q25 = data["Close_NVDA"].quantile(0.25)
q50 = data["Close_NVDA"].quantile(0.50)
q75 = data["Close_NVDA"].quantile(0.75)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data["Close_NVDA"], color='blue', label="Cena zamknięcia")
plt.xlabel("Data")
plt.ylabel("Cena zamknięcia (USD)")
plt.title("Cena zamknięcia akcji NVDA w czasie")

plt.axhline(y=q25, color='red', linestyle='--', label=f'Q25: {q25:.2f}')
plt.axhline(y=q50, color='green', linestyle='--', label=f'Q50 (Mediana): {q50:.2f}')
plt.axhline(y=q75, color='orange', linestyle='--', label=f'Q75: {q75:.2f}')

plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# Widzimy rosnący trend, co możemy potwierdzić testem Manna-Kendalla. Jest to test nieparametryczny test statystyczny (nie zakłada żadnego konkretnego rozkładu danych), który służy do wykrywania trendu w szeregach czasowych.
# - Jeśli wartość statystyki testu jest dodatnia, oznacza to, że dane mają tendencję rosnącą.
# - Jeśli wartość statystyki testu jest ujemna, oznacza to, że dane mają tendencję malejącą.
# - Jeśli wartość testu jest bliska zeru, to oznacza, że w danych nie widać wyraźnego trendu.

# %%
trend_test = mk.original_test(data['Close_NVDA'])
print(f"Trend: {trend_test.trend}")
print(f"Nachylenie: {trend_test.slope}")
print(f"p-value: {trend_test.p}")

# %% [markdown]
# Test potwierdza obecność trednu rosnącego. Zróżnicowanie danych może pomóc w usunięciu trendu i sprawieniu, by szereg czasowy stał się stacjonarny (czyli jego statystyki (średnia, wariancja, kowariancja) nie zmieniały się w czasie).

# %% [markdown]
# ## Zróżnicowanie szeregu czasowego Close_NVDA
# Nasze dane są nieciągłe (brakuje weekendów i świąt), więc różnicowanie będzie dotyczyło tylko dni handlowych. Uzupełnianie brakujących dni sztucznymi wartościami (np. przez forward fill) może wprowadzić fałszywe informacje do analizy. W rezultacie uwzględniamy tylko rzeczywiste zmiany cen w dostępnych datach.

# %%
data["diff"] = data["Close_NVDA"].diff()
data = data.dropna(subset=["diff"])

# %% [markdown]
# Sprawdzimy jak zmieniły się dane na wykresie:

# %%
plt.figure(figsize=(10, 6))
plt.plot(data.index, data["diff"], color='blue', label="Zróżnicowana cena zamknięcia")
plt.xlabel("Data")
plt.ylabel("Zmiana ceny (USD)")
plt.title("Zmiana ceny zamknięcia akcji NVDA w czasie")

plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# Na wykresie nie zauważamy wyraźnego trendu. Sprawdzimy to za pomocą testu Manna-Kendalla:

# %%
trend_test = mk.original_test(data["diff"])
print(f"Trend: {trend_test.trend}")
print(f"Nachylenie: {trend_test.slope}")
print(f"p-value: {trend_test.p}")

# %% [markdown]
# Test potwierdza pozbycie się trendu. Możemy również przetestować stacjonarność zróżnicowanego szeregu za pomocą rozszerzonego testu Dickey-Fullera (ADF), który sprawdza, czy szereg ma jednostkowy pierwiastek (co sugerowałoby, że jest niestacjonarny). Jeśli p-value testu ADF będzie małe (np. poniżej 0.05), oznacza to, że po zróżnicowaniu szereg stał się stacjonarny:

# %%
result = adfuller(data["diff"].dropna())
print("p-value:", result[1])

# %% [markdown]
# Wartość p-value ($1.96 \times 10^{-28}$) jest bardzo bliska zeru, co sugeruje, że szereg czasowy jest stacjonarny. Możemy sprawdzić wartości odstające:

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(data["diff"], color='lightblue')
plt.title("Wartości odstające w zróżnicowanej cenie zamknięcia akcji NVDA")
plt.xlabel("Zmiana ceny (diff)")
plt.grid(True)
plt.show()

# %% [markdown]
# - Mediana (środkowa linia w pudełku) różnicy cen zamknięcia akcji NVDA wynosi około 0, co sugeruje, że większość zmian cen oscyluje wokół tej wartości.
# - Pudełko obejmuje zakres od pierwszego do trzeciego kwartylu, co oznacza, że 50% danych znajduje się w przedziale od około -1 do 1.
# - Wykres pokazuje kilka wartości odstających, które są zaznaczone jako pojedyncze punkty powyżej i poniżej wąsów. Wartości te są znacznie wyższe lub niższe niż reszta danych, co może wskazywać na nietypowe zmiany cen.
# - Wąsy rozciągają się od dolnego do górnego limitu, co pokazuje pełny zakres danych z wyłączeniem wartości odstających.
#
# Możemy również wyświetlić statystyki zróżnicowanego szeregu:

# %%
print(data["diff"].describe())

# %% [markdown]
# - Średnia zmiana ceny jest dodatnia (0.141 USD), co sugeruje, że akcje NVDA generalnie rosły w badanym okresie.
# - Wysokie odchylenie standardowe (1.02 USD) oznacza, że ceny zmieniały się znacznie z dnia na dzień, co wskazuje na dużą zmienność.
# - Rozkład jest asymetryczny, ponieważ mediana (0.139 USD) jest bliska średniej, ale 25. percentyl jest ujemny (-0.443 USD), co sugeruje, że niektóre dni miały większe straty niż zyski.
# - Wartości ekstremalne (np. maksymalna zmiana 7.442 USD) mogą wskazywać na duże wahania cen w okresach dużych wzrostów.

# %% [markdown]
# ## Analiza rozkładu
# Za pomocą histogramu możemy zobaczyć, jak rozkładają się wartości w zróżnicowanym szeregu:

# %%
plt.figure(figsize=(10, 6))
data["diff"].hist(bins=50, edgecolor='k', color='orange')
plt.title("Histogram zróżnicowanej ceny zamknięcia akcji NVDA")
plt.xlabel("Zmiana ceny (diff)")
plt.ylabel("Częstotliwość")
plt.grid(True)
plt.show()

# %% [markdown]
# - Najwięcej zmian ceny zamknięcia akcji NVDA oscyluje wokół wartości 0, co oznacza, że najczęściej cena zamknięcia nie zmieniała się znacząco.
# - Większość zmian ceny mieści się w przedziale od -2 do 2, co sugeruje, że zmiany ceny zamknięcia akcji NVDA są zazwyczaj niewielkie.
# - Istnieje kilka przypadków, gdzie zmiana ceny była większa niż 4, ale są one rzadkie.
# - Histogram jest asymetryczny, z większą liczbą przypadków po stronie dodatnich zmian ceny.
#
# Na podstawie histogramu, nasze dane mogą mieć rozkład normalny (rozkład Gaussa), ponieważ większość wartości skupia się wokół średniej (0), a liczba obserwacji maleje symetrycznie w miarę oddalania się od średniej. Jednakże, po prawej stronie widzimy pewne odchylenia od idealnego rozkładu normalnego (wartości skrajne).

# %%
plt.figure(figsize=(10, 6))
plt.hist(data["diff"], bins=50, density=True, color='orange', edgecolor='k')

# dopasowanie rozkładu normalnego
mu, std = norm.fit(data["diff"])

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title(f"Dopasowanie rozkładu normalnego: mu = {mu:.2f},  std = {std:.2f}")
plt.xlabel("Zmiana ceny (diff)")
plt.ylabel("Gęstość prawdopodobieństwa")
plt.grid(True)
plt.show()

# %% [markdown]
# wnioski...
# - gruby ogon z prawej strony...

# %% [markdown]
# ## Kalkulacja VaR
# ### a) Obliczanie VaR przy założeniu rozkładu normalnego

# %% [markdown]
# Policzymy średnią ($\mu$) i odchylenie standardowe ($\sigma$):

# %%
mu = data["diff"].mean()
sigma = data["diff"].std()
print(f"Mu: {mu}")
print(f"Sigma: {sigma}")

# %% [markdown]
# Definiujemy poziom ufności $\alpha$ 5% i 1%:

# %%
alpha_95 = 0.05
alpha_99 = 0.01

# %% [markdown]
# Obliczamy Value at Risk czyli wielkość najgorszej straty przy poziomie ufności $\alpha$ 5% i 1%:

# %%
VaR_95 = norm.ppf(alpha_95, loc=mu, scale=sigma)
VaR_99 = norm.ppf(alpha_99, loc=mu, scale=sigma)

# %%
print(f"VaR 95% (rozkład normalny): {VaR_95}")
print(f"VaR 99% (rozkład normalny): {VaR_99}")

# %% [markdown]
# ### b) Obliczanie VaR dla rozkładu historycznego

# %% [markdown]
# Dla podejścia historycznego wystarczy policzyć odpowiedni kwantyl:

# %%
VaR_95_hist = data["diff"].quantile(alpha_95)
VaR_99_hist = data["diff"].quantile(alpha_99)

# %%
print(f"VaR 95% (rozkład historyczny): {VaR_95_hist}")
print(f"VaR 99% (rozkład historyczny): {VaR_99_hist}")

# %% [markdown]
# Możemy porównać VaR parametryczny (z rozkładu normalnego) i VaR historyczny za pomocą wykresu:

# %%
plt.figure(figsize=(10, 6))
data["diff"].hist(bins=50, edgecolor='k', color='orange')
plt.title("Histogram zróżnicowanej ceny zamknięcia akcji NVDA")
plt.xlabel("Zmiana ceny (diff)")
plt.ylabel("Częstotliwość")

# dodanie linii VaR dla podejścia historycznego
plt.axvline(VaR_95_hist, color='black', linestyle='--', linewidth=1, label=f'VaR 95% Historyczny: {VaR_95_hist:.2f}')
plt.axvline(VaR_99_hist, color='black', linestyle='-', linewidth=1, label=f'VaR 99% Historyczny: {VaR_99_hist:.2f}')

# dodanie linii VaR dla podejścia parametrycznego
plt.axvline(VaR_95, color='red', linestyle='--', linewidth=1, label=f'VaR 95% Parametryczny: {VaR_95:.2f}')
plt.axvline(VaR_99, color='red', linestyle='-', linewidth=1, label=f'VaR 99% Parametryczny: {VaR_99:.2f}')

plt.legend()
plt.show()

# %% [markdown]
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

# zysk lub starta

# dane do zyskow i strat, mamy powiedziec co chcemy liczyc, cos tam z var, test na rozklad normalny, komentarzy duzo co widzimy na wykresach
