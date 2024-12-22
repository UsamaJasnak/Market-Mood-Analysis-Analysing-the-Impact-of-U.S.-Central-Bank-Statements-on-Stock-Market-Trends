library('forecast')
library('readxl')
library('ggplot2')

## SP500

df = read_excel('..\\r_scripts\\SP500.xlsx', sheet=1)
Y <- ts(df[, ncol(df)], frequency=7)
X <- as.matrix(df[, -ncol(df)])

X_first4 <- X[, 2:5]
X_rest <- X[, -(1:5)]

# PCA
pca_result <- prcomp(X_rest, scale.=TRUE)
pca_components <- pca_result$x[, 1:4]
X_final <- cbind(X_first4, pca_components)

# Modelling
model <- auto.arima(Y, xreg=X_final, approximation=TRUE,
                    stepwise=TRUE, trace=TRUE)

summary(model)
checkresiduals(model)

## BAC

df = read_excel('..\\r_scripts\\BAC.xlsx', sheet=1)
Y <- ts(df[, ncol(df)], frequency=7)
X <- as.matrix(df[, -ncol(df)])

X_first4 <- X[, 2:5]
X_rest <- X[, -(1:5)]

# PCA
pca_result <- prcomp(X_rest, scale.=TRUE)
pca_components <- pca_result$x[, 1:4]
X_final <- cbind(X_first4, pca_components)

# Modelling
model <- auto.arima(Y, xreg=X_final, approximation=TRUE,
                    stepwise=TRUE, trace=TRUE)

summary(model)
checkresiduals(model)
