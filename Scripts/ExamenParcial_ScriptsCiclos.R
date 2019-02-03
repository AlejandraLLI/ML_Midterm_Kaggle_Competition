setwd("E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/Scripts")

#---- Cargamos paquetes----
# install.packages("tidyverse")
# install.packages("data.table")
# install.packages("glmnet")
library(tidyverse)
library(data.table)
library(glmnet)
library(keras)

#---- Leemos los datos de entrenamiento ----
entrena<-read_csv('E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/datos/train.csv')
entrena2<-entrena

entrena <- entrena %>%
  mutate(estado=ifelse(estado=='abierta',0,1),
         hour= hour(hora), 
         minutes= minute(hora),
         hora=as.numeric(hora))

train<-entrena
#---- Leemos los datos de prueba ----
test<-read_csv('E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/datos/test.csv')
test2<-test

test <- test %>%
  mutate(hour= hour(hora), 
         minutes= minute(hora),
         hora=as.numeric(hora))



#---- Mod20 LASSO sin estandarizar, con la fecha completa (sin hora y minutos)----

# Extraemos las enradas
set.seed(12345)
x_mod20<-as.matrix(train[,c(1:38000,38002)])

# Corremos un modelo glm.
mod20<-cv.glmnet(x=x_mod20,y=train$estado,
                 alpha=1,
                 family='binomial',
                 intercept=FALSE,
                 parallel=TRUE,
                 nfolds=10,
                 lambda=exp(seq(-12,2,1)))

plot(mod20)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod20<-predict(mod20,newx=x_mod20,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod20<-ifelse(proba_entrena_mod20>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod20<-mod20$cvm[mod20$lambda==mod20$lambda.1se]

# Obtenemos la matriz de confusión de entrenamiento
table(proba_entrena_mod20>0.5,train$estado)
prop.table(table(proba_entrena_mod20>0.5,train$estado),2)

# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod20<-as.matrix(test[,1:38001])
proba_prueba_mod20<-predict(mod20,newx=x_prueba_mod20,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod20<-ifelse(proba_prueba_mod20>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod20,"E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/Resultados/Result_prueba_mod20.csv")


#---- Mod21 RIDGE sin estandarizar, con la fecha completa (sin hora y minutos)----

# Extraemos las enradas
set.seed(12345)
x_mod21<-as.matrix(train[,c(1:38000,38002)])

# Corremos un modelo glm.
mod21<-cv.glmnet(x=x_mod21,y=train$estado,
                 alpha=0,
                 family='binomial',
                 intercept=FALSE,
                 parallel=TRUE,
                 nfolds=10,
                 lambda=exp(seq(-12,2,1)))

plot(mod21)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod21<-predict(mod21,newx=x_mod21,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod21<-ifelse(proba_entrena_mod21>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod21<-mod21$cvm[mod21$lambda==mod21$lambda.1se]

# Obtenemos la matriz de confusión de entrenamiento
table(proba_entrena_mod21>0.5,train$estado)
prop.table(table(proba_entrena_mod21>0.5,train$estado),2)

# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod21<-as.matrix(test[,1:38001])
proba_prueba_mod21<-predict(mod21,newx=x_prueba_mod21,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod21<-ifelse(proba_prueba_mod21>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod21,"E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/Resultados/Result_prueba_mod21.csv")

