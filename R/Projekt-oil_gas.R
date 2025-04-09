library(dplyr)
library(gamlss)
library(ggplot2)
library(ggExtra)
library(tidyr)
library(fitdistrplus)
library(copula)
library(MASS)

# Preprocessing
data <- read.csv("./notebooks/data/oil_and_gas.csv")
data$Date <- as.Date(data$Date)
head(data)
unique(data$Symbol)

data <- data[, c("Symbol", "Date", "Close")]
head(data)

data <- data %>%
  pivot_wider(names_from = Symbol, values_from = Close)
colnames(data) <- c("Date", "Oil", "1", "Gas", "2")
data <- data[, c("Date", "Oil", "Gas")]
data[1:10,]
sum(is.na(data)) # 176
data <- na.omit(data)
sum(is.na(data)) # 0

str(data)
summary(data)

# Wykres
plot(data$Date, data$Oil, type = "l", col = "blue", ylim=c(0,150),
     xlab = "Data", ylab = "Cena zamknięcia [USD]", 
     main = "Ceny zamknięcia gazu i ropy w czasie")
lines(data$Date, data$Gas, col = "orange")
grid(nx = NULL, ny = NULL, lty=1, col = "gray",lwd = 2)
legend("topleft", legend = c("Ropa", "Gaz"), col = c("blue", "orange"), lty = 1)

# Zróżnicowanie szeregów
data$diff_oil <- c(NA, diff(data$Oil))
data$diff_gas <- c(NA, diff(data$Gas))
head(data)

data <- data %>% drop_na()
head(data)
summary(data[, c("diff_oil","diff_gas")])

# Wykres różnic
plot(data$Date, data$diff_oil, type = "l", col = "blue",
     xlab = "Data", ylab = "Różnica [USD]", 
     main = "Różnice dzienne cen ropy i gazu")
lines(data$Date, data$diff_gas, col = "orange")
grid(nx = NULL, ny = NULL, lty=1, col = "gray",lwd = 2)
legend("topleft", legend = c("Ropa", "Gaz"), col = c("blue", "orange"), lty = 1)

# Histogramy
par(mfrow=c(1,2))
hist(data$diff_oil, prob=T, ylim=c(0,1), xlim=c(-15,10), main="Rozkład zróżnicowanych cen ropy",
     xlab="Zmiana ceny [USD]", col="blue", breaks=50)
hist(data$diff_gas, prob=T, ylim=c(0,3.7), xlim=c(-3,3), main="Rozkład zróżnicowanych cen gazu",
     xlab="Zmiana ceny [USD]", col="orange", breaks=50)

X1 <- data$diff_oil
X2 <- data$diff_gas
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
fit.1$family # JSUo "Johnson SU original"
fit.2$family # JSUo "Johnson SU original"

par1 <- c(fit.1$mu, fit.1$sigma, fit.1$nu, fit.1$tau)
par2 <- c(fit.2$mu, fit.2$sigma, fit.2$nu, fit.2$tau)
par1; par2

U <- data.frame(pJSUo(X1, par1), pJSUo(X2, par2))
colnames(U) <- c("U1","U2")
head(U)

p1 <- ggplot(X, aes(X1, X2)) + geom_point()
ph1 <- ggMarginal(p1, type="histogram") 
p2 <- ggplot(U, aes(U1, U2)) + geom_point()
ph2 <- ggMarginal(p2, type="histogram") 
cowplot::plot_grid(ph1, ph2, ncol=1, nrow=2)

cop.par <- fitCopula(copula=normalCopula(), data=U)
cop.par # rho 0.49

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
cop.npar # rho 0.22

par(mfrow=c(2,1))
persp(normalCopula(dim=2,cop.npar@estimate),pCopula, col=8)
persp(normalCopula(dim=2,cop.npar@estimate),dCopula, col=8)

