#---- Cargamos paquetes----
# install.packages("tidyverse")
# install.packages("data.table")
# install.packages("glmnet")
library(tidyverse)
library(data.table)
library(glmnet)

#---- Leemos los datos de entrenamiento ----
train<-read_csv('train.csv')
train2<-train

# train <- train %>%
#   mutate(estado=ifelse(estado=='abierto',1,0),hour= hour(hora), minutes= minute(hora))%>%
#   select(-hora)

train <- train %>%
  mutate(estado=ifelse(estado=='abierta',1,0),
         hour= hour(hora), 
         minutes= minute(hora)) %>%
  select(-hora)


#---- Leemos los datos de prueba ----
test<-read_csv('test.csv')
test2<-test

test <- test %>%
  mutate(hour= hour(hora), 
         minutes= minute(hora)) %>%
  select(-hora)


#---- Modelos ----
# Regresión Logística
# mod1<-glm(estado~.,data=train,family='binomial')

# Regresión Lasso
x_mod2<-as.matrix(train[,1:38000])

mod2<-glmnet(x=x_mod2,y=train$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Extraemos los coeficientes distintos de cero
coef_mod2<-data.frame(pixel=which(coef(mod2)!=0),coef=coef(mod2)[which(coef(mod2)!=0)])

# Obtenemos todos los coeficientes del modelo
coef_mod2_lasso<-vector("numeric",ncol(x_mod2))
coef_mod2_lasso[coef_mod2$pixel]<-coef_mod2$coef

# Creamos la función logística
h<-function(x){
  1/(1+exp(-x))
}

# --- Error Entrenamiento y Clasificación Entrenamiento----

# Obtenemos las probabilidades de clase
proba_entrena_mod2<-h(x_mod2%*%as.numeric(coef_mod2_lasso))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod2<-ifelse(proba_entrena_mod2>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod2>0.5,train$estado)


# --- Error Prueba y Clasificador Prueba----

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod2<-as.matrix(test[,1:38000])
proba_prueba_mod2<-h(x_prueba_mod2%*%as.numeric(coef_mod2_lasso))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod2<-ifelse(proba_prueba_mod2>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod2,"Result_prueba_mod2.csv")
