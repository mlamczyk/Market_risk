library(dplyr)
library(gamlss)
library(ggplot2)
library(ggExtra)
library(tidyr)
library(fitdistrplus)
library(copula)
library(MASS)
library(reticulate)

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

### Dopasowanie kopuły Gaussa ###

# Podejście parametryczne
fit.1 <- fitDist(X1, type="realline")
fit.2 <- fitDist(X2, type="realline")
fit.1$family # JSU "Johnson SU"
fit.2$family # SEP 2 "Skew Exponential Power type 2"

par1 <- c(fit.1$mu, fit.1$sigma, fit.1$nu, fit.1$tau)
par2 <- c(fit.2$mu, fit.2$sigma, fit.2$nu, fit.2$tau)
par1; par2

U <- data.frame(pJSU(X1, par1), pSEP2(X2, par2))
colnames(U) <- c("U1","U2")
head(U)

p1 <- ggplot(X, aes(X1, X2)) + geom_point()
ph1 <- ggMarginal(p1, type="histogram") 
p2 <- ggplot(U, aes(U1, U2)) + geom_point()
ph2 <- ggMarginal(p2, type="histogram") 
cowplot::plot_grid(ph1, ph2, ncol=1, nrow=2)

cop.par <- fitCopula(copula=normalCopula(), data=U)
cop.par # rho 0.34

par(mfrow=c(2,1))
persp(normalCopula(dim=2,cop.par@estimate), pCopula, col=8)
persp(normalCopula(dim=2,cop.par@estimate), dCopula, col=8)

# Podejście nieparametryczne
V <- pobs(X)
head(V)
V <- data.frame(V1=V[,1],V2=V[,2])
head(V)

p3 <- ggplot(V, aes(V1,V2))+geom_point()
ph3 <- ggMarginal(p3, type="histogram") 
cowplot::plot_grid(ph1, ph2, ph3, ncol = 1, nrow = 3)

cop.npar=fitCopula(copula=normalCopula(), data=V)
cop.npar # rho 0.64

par(mfrow=c(2,1))
persp(normalCopula(dim=2,cop.npar@estimate),pCopula, col=8)
persp(normalCopula(dim=2,cop.npar@estimate),dCopula, col=8)

