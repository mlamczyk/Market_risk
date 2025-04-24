library(dplyr)
library(gamlss)
library(ggplot2)
library(ggExtra)
library(tidyr)
library(fitdistrplus)
library(copula)
library(MASS)
library(reticulate)
library(MVN)

# Importowanie biblioteki yfinance z Pythona
yf <- import("yfinance")

# Pobieranie danych
data <- yf$download(c("NVDA", "AMD"), start = "2020-01-03", end = "2025-01-01")
colnames(data) <- c("AMD_Close", "NVDA_Close", "AMD_Open", "NVDA_Open",
                    "AMD_High", "NVDA_High", "AMD_Low", "NVDA_Low", "AMD_Volume", "NVDA_Volume")

data <- data[, c("NVDA_Close", "AMD_Close")]
data$Date <- as.Date(rownames(data))
head(data)

str(data)
summary(data)

# Wykres
plot(data$Date, data$NVDA_Close, type = "l", col = "green", ylim=c(0,210),
     xlab = "Data", ylab = "Cena zamknięcia [USD]", 
     main = "Ceny zamknięcia akcji NVDA i AMD")
lines(data$Date, data$AMD_Close, col = "red")
grid(nx = NULL, ny = NULL, lty=1, col = "gray",lwd = 2)
legend("topleft", legend = c("NVDA", "AMD"), col = c("green", "red"), lty = 1)

# Zróżnicowanie szeregów
data$diff_NVDA <- c(NA, diff(data$NVDA_Close))
data$diff_AMD <- c(NA, diff(data$AMD_Close))
head(data)

data <- data %>% drop_na()
head(data)
summary(data[, c("diff_NVDA","diff_AMD")])

# Wykres różnic
plot(data$Date, data$diff_AMD, type = "l", col = "red",
     xlab = "Data", ylab = "Różnica [USD]", 
     main = "Różnice dzienne dla NVDA i AMD")
lines(data$Date, data$diff_NVDA, col = "green")
grid(nx = NULL, ny = NULL, lty=1, col = "gray",lwd = 2)
legend("topleft", legend = c("NVDA", "AMD"), col = c("green", "red"), lty = 1)

# Histogramy
par(mfrow=c(1,2))
hist(data$diff_NVDA, prob=T, ylim=c(0,0.7), xlim=c(-15,15), main="Rozkład zróżnicowanych cen NVDA",
     xlab="Zmiana ceny [USD]", col="green", breaks=50)
hist(data$diff_AMD, prob=T, ylim=c(0,0.3), xlim=c(-20,20), main="Rozkład zróżnicowanych cen AMD",
     xlab="Zmiana ceny [USD]", col="red", breaks=50)

X1 <- data$diff_NVDA
X2 <- data$diff_AMD
X <- data.frame(X1=X1, X2=X2)
X <- X[complete.cases(X), ]
dim(X)

# Wykres rozrzutu z histogramami
p <- ggplot(X, aes(X1,X2)) + geom_point()
ggMarginal(p, type="histogram")


# Podejście parametryczne - zakładamy, że dane pochodzą z konkretnych rozkładów

### Dopasowanie rozkładów brzegowych ###

fit.1 <- fitDist(X1, type="realline")
fit.2 <- fitDist(X2, type="realline")
fit.1$family # JSU "Johnson SU"
fit.2$family # SEP 2 "Skew Exponential Power type 2"

# Parametry dopasowanych rozkładów
par1 <- c(fit.1$mu, fit.1$sigma, fit.1$nu, fit.1$tau)
par2 <- c(fit.2$mu, fit.2$sigma, fit.2$nu, fit.2$tau)
par1; par2

# Przekształcenie danych do przedziału (0,1)
U <- data.frame(pJSU(X1, par1), pSEP2(X2, par2))
colnames(U) <- c("U1","U2")
U <- as.matrix(U)
head(U)

p1 <- ggplot(X, aes(X1, X2)) + geom_point()
ph1 <- ggMarginal(p1, type="histogram") 
p2 <- ggplot(U, aes(U1, U2)) + geom_point()
ph2 <- ggMarginal(p2, type="histogram") 
cowplot::plot_grid(ph1, ph2, ncol=1, nrow=2)

### Dopasowanie kopuły Gaussa ###

cop.gauss <- fitCopula(copula=normalCopula(), data=U)
cop.gauss # rho 0.34

aic.gauss = AIC(cop.gauss)
bic.gauss = BIC(cop.gauss)
loglik.gauss = logLik(cop.gauss)

par(mfrow=c(2,1))
persp(normalCopula(dim=2, cop.gauss@estimate), pCopula, col=8)
persp(normalCopula(dim=2, cop.gauss@estimate), dCopula, col=8)

### Dopasowanie kopuły t-Studenta ###

#cop.tstudent <- fitCopula(copula=tCopula(), data=U)

start_param <- c(0.3, 5)  # rho=0.3, df=5
copula_t <- tCopula(dim = 2, df.fixed = FALSE)
cop.tstudent <- fitCopula(
  copula = copula_t,
  data = U,
  method = "ml",
  start = start_param
)
cop.tstudent # rho 0.31, df 24

aic.tstudent = AIC(cop.tstudent)
bic.tstudent = BIC(cop.tstudent)
loglik.tstudent = logLik(cop.tstudent)

par(mfrow=c(2,1))
persp(tCopula(dim=2, param=cop.tstudent@estimate[1], df=floor(cop.tstudent@estimate[2])), pCopula, col=8)
persp(tCopula(dim=2, param=cop.tstudent@estimate[1], df=floor(cop.tstudent@estimate[2])), dCopula, col=8)

### Dopasowanie kopuły Claytona ###

cop.clayton <- fitCopula(copula=claytonCopula(), data=U)
cop.clayton # alpha 1.95

aic.clayton = AIC(cop.clayton)
bic.clayton = BIC(cop.clayton)
loglik.clayton = logLik(cop.clayton)

par(mfrow=c(2,1))
persp(claytonCopula(dim=2, cop.clayton@estimate), pCopula, col=8)
persp(claytonCopula(dim=2, cop.clayton@estimate), dCopula, col=8)

### Dopasowanie kopuły Gumbela ###

cop.gumbel <- fitCopula(copula=gumbelCopula(), data=U)
cop.gumbel # alpha 1.27

aic.gumbel = AIC(cop.gumbel)
bic.gumbel = BIC(cop.gumbel)
loglik.gumbel = logLik(cop.gumbel)

par(mfrow=c(2,1))
persp(gumbelCopula(dim=2, cop.gumbel@estimate), pCopula, col=8)
persp(gumbelCopula(dim=2, cop.gumbel@estimate), dCopula, col=8)

### Dopasowanie kopuły Franka ###

cop.frank <- fitCopula(copula=frankCopula(), data=U)
cop.frank # alpha 4.24

aic.frank = AIC(cop.frank)
bic.frank = BIC(cop.frank)
loglik.frank = logLik(cop.frank)

par(mfrow=c(2,1))
persp(frankCopula(dim=2, cop.frank@estimate), pCopula, col=8)
persp(frankCopula(dim=2, cop.frank@estimate), dCopula, col=8)

### Ocena doapasowania ###

results <- data.frame (
  Copula = c("Gauss", "t-Student", "Clayton", "Gumbel", "Frank"),
  AIC = c(aic.gauss, aic.tstudent, aic.clayton, aic.gumbel, aic.frank),
  BIC = c(bic.gauss, bic.tstudent, bic.clayton, bic.gumbel, bic.frank),
  LogLik = c(loglik.gauss, loglik.tstudent, loglik.clayton, loglik.gumbel, loglik.frank)
)

# Sortujemy wyniki roznąco według wartości BIC - im niższa tym lepiej
results_sorted <- results[order(results$AIC),]
results_sorted

# Najlepsza kopuła
results_sorted[1,1] # t-Student

# ----------------------------------------------------------------
# Dla sprawdzenia - najlepsza kopuła z VineCopula
library(VineCopula)
cop <- BiCopSelect(U[,1],U[,2])
cop # Bivariate copula: t (par = 0.32, par2 = 17.57, tau = 0.21)
# czyli też kopuła t-Studenta

# Porównanie konturu 'empirycznego' i teoretycznego (kopuły)
par(mfrow=c(2,1))
# kontur empiryczny - rzeczywisty rozkład zależności między zmiennymi
BiCopKDE(U[,1],U[,2], type="contour", main="Empiryczny kontur (KDE)")
# kontur teoretyczny gęstości kopuły
contour(cop, main="Teoretyczny kontur (kopuła)")
# ---------------------------------------------------------------

### Test Mardia - wielowymiarowa normalność ###

mvn(data=X, mvnTest="mardia")$multivariateNormality

# Skośność wielowymiarowa (skewness) – bada, czy dane są symetryczne
# Kurtoza wielowymiarowa (kurtosis) – bada, czy rozkład ma odpowiednią „grubość ogonów”

# p-value w obu przypadkach jest bardzo niskie, więc odrzucamy hipotezę zerową
# o wielowymiarowej normalności - dane nie są wielowymiarowo normalne


### Próbki z rozkładów brzegowych ###

# Obliczanie VaR dla portfela dla różnych wartości beta

# Optymalne beta i minimalne VaR


### Próbki z kopuły t-Studenta ###

# Obliczanie VaR dla portfela dla różnych wartości beta

# Optymalne beta i minimalne VaR


### Porównanie wyników i wnioski ###

