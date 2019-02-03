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
  mutate(estado=ifelse(estado=='abierta',0,1),
         hour= hour(hora), 
         minutes= minute(hora)) %>%
  select(-hora)

#---- Estandarizamos los datos de entrenamiento----#
train$id<-1:nrow(train)

train_MeanSD<-train %>% 
  gather(Variable, Valor, -estado) %>% # Apilamos
  group_by(Variable) %>% # Agrupamos por variable
  summarise(media=mean(Valor), de=sd(Valor)) # calcula media y desviación.

# Función normalizar
normalizar_train<-function(datos,datos_MeanSD){
  datos%>%
    gather(Variable,Valor,-c(estado,id)) %>% #apilamos los datos
    left_join(datos_MeanSD) %>% # se pega la media y desv correspondiente
    mutate(Valor_norm=(Valor-media)/de) %>% #se estandariza
    select(id,estado,Variable,Valor_norm) %>% #se seleccionana las columnas 
    spread(Variable,Valor_norm) %>%# se desapilan los datos.
    select(-id)#Se elimina la columna id
    
}

# Se normalizan los datos de entranmiento.
train_norm<-normalizar_train(train,train_MeanSD)

# Se reordenan las columnas
train<-train%>%select(-id)
cols<-match(colnames(train),colnames(train_norm))
train_norm<-train_norm[,cols]

#---- Leemos los datos de prueba ----
test<-read_csv('test.csv')
test2<-test

test <- test %>%
  mutate(hour= hour(hora), 
         minutes= minute(hora)) %>%
  select(-hora)


#---- Estandarizamos los datos de prueba----
test$id<-1:nrow(test)

# Función normalizar
normalizar_test<-function(datos,datos_MeanSD){
  datos%>%
    gather(Variable,Valor,-c(id)) %>% #apilamos los datos
    left_join(datos_MeanSD) %>% # se pega la media y desv correspondiente
    mutate(Valor_norm=(Valor-media)/de) %>% #se estandariza
    select(id,Variable,Valor_norm) %>% #se seleccionana las columnas 
    spread(Variable,Valor_norm) %>%# se desapilan los datos.
    select(-id)#Se elimina la columna id
  
}

# Se normalizan los datos de entranmiento.
test_norm<-normalizar_test(test,train_MeanSD)

# Se reordenan las columnas
test<-test %>% select(-id)
cols<-match(colnames(test),colnames(test_norm))
test_norm<-test_norm[,cols]

#---- Funciones ----
# Creamos la función logística
h<-function(x){
  1/(1+exp(-x))
}

# Perdida logística
devianza<-function(estado,probs){
  
  # No de observaciones
  n<-length(estado)
  
  # Pérdida logística
  mean(-(estado*log(probs)+(1-estado)*log(1-probs)))
  
}


#---- Modelo 1: Reg Logística ----
# Notas: Este modelo no funciona, truena por memoria.
# Regresión Logística
# mod1<-glm(estado~.,data=train,family='binomial')

#---- Modelo 2: Reg Lasso (Sin Horas ni Minuos, sin estandarizar)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod2<-as.matrix(train[,1:38000])

# Corremos un modelo glm.
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

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod2<-h(x_mod2%*%as.numeric(coef_mod2_lasso))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod2<-ifelse(proba_entrena_mod2>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod2>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod2<-as.matrix(test[,1:38000])
proba_prueba_mod2<-h(x_prueba_mod2%*%as.numeric(coef_mod2_lasso))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod2<-ifelse(proba_prueba_mod2>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod2,"Result_prueba_mod2.csv")

#---- Modelo 3: Reg Lasso (Con Horas y Minuos, sin estandarizar)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod3<-as.matrix(train[,c(1:38000,38002:ncol(train))])

# Corremos un modelo glm.
mod3<-glmnet(x=x_mod3,y=train$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Extraemos los coeficientes distintos de cero
coef_mod3<-data.frame(pixel=which(coef(mod3)!=0),coef=coef(mod3)[which(coef(mod3)!=0)])

# Obtenemos todos los coeficientes del modelo
coef_mod3_lasso<-vector("numeric",ncol(x_mod3))
coef_mod3_lasso[coef_mod3$pixel]<-coef_mod3$coef

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod3<-h(x_mod3%*%as.numeric(coef_mod3_lasso))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod3<-ifelse(proba_entrena_mod3>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod3>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod3<-as.matrix(test)
proba_prueba_mod3<-h(x_prueba_mod3%*%as.numeric(coef_mod3_lasso))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod3<-ifelse(proba_prueba_mod3>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod3,"Result_prueba_mod3.csv")

#---- Modelo 4: Reg Lasso (Sin Horas ni Minuos, estandarizado)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod4<-as.matrix(train_norm[,1:38000])

# Corremos un modelo glm.
mod4<-glmnet(x=x_mod4,y=train_norm$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Extraemos los coeficientes distintos de cero
coef_mod4<-data.frame(pixel=which(coef(mod4)!=0),coef=coef(mod4)[which(coef(mod4)!=0)])

# Obtenemos todos los coeficientes del modelo
coef_mod4_lasso<-vector("numeric",ncol(x_mod4))
coef_mod4_lasso[coef_mod4$pixel]<-coef_mod4$coef

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod4<-h(x_mod4%*%as.numeric(coef_mod4_lasso))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod4<-ifelse(proba_entrena_mod4>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod4>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod4<-as.matrix(test_norm[,1:38000])
proba_prueba_mod4<-h(x_prueba_mod4%*%as.numeric(coef_mod4_lasso))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod4<-ifelse(proba_prueba_mod4>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod4,"Result_prueba_mod4.csv")

#---- Modelo 5: Reg Lasso (Con Horas y Minuos, estandarizado)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod5<-as.matrix(train_norm[,c(1:38000,38002:ncol(train_norm))])

# Corremos un modelo glm.
mod5<-glmnet(x=x_mod5,y=train_norm$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Extraemos los coeficientes distintos de cero
coef_mod5<-data.frame(pixel=which(coef(mod5)!=0),coef=coef(mod5)[which(coef(mod5)!=0)])

# Obtenemos todos los coeficientes del modelo
coef_mod5_lasso<-vector("numeric",ncol(x_mod5))
coef_mod5_lasso[coef_mod5$pixel]<-coef_mod5$coef

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod5<-h(x_mod5%*%as.numeric(coef_mod5_lasso))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod5<-ifelse(proba_entrena_mod5>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod5>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod5<-as.matrix(test_norm)
proba_prueba_mod5<-h(x_prueba_mod5%*%as.numeric(coef_mod5_lasso))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod5<-ifelse(proba_prueba_mod5>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod5,"Result_prueba_mod5.csv")

#---- Modelo 6: Reg Ridge (Sin Horas ni Minuos, sin estandarizar)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod6<-as.matrix(train[,1:38000])

# Corremos un modelo glm.
mod6<-glmnet(x=x_mod6,y=train$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Obtenemos todos los coeficientes del modelo
coef_mod6_Ridge<-coef(mod6)[-1]

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod6<-h(x_mod6%*%as.numeric(coef_mod6_Ridge))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod6<-ifelse(proba_entrena_mod6>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod6>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod6<-as.matrix(test[,1:38000])
proba_prueba_mod6<-h(x_prueba_mod6%*%as.numeric(coef_mod6_Ridge))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod6<-ifelse(proba_prueba_mod6>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod6,"Result_prueba_mod6.csv")

#---- Modelo 7: Reg Ridge (Con Horas y Minuos, sin estandarizar)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod7<-as.matrix(train[,c(1:38000,38002:ncol(train))])

# Corremos un modelo glm.
mod7<-glmnet(x=x_mod7,y=train$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Obtenemos todos los coeficientes del modelo
coef_mod7_Ridge<-coef(mod7)[-1]

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod7<-h(x_mod7%*%as.numeric(coef_mod7_Ridge))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod7<-ifelse(proba_entrena_mod7>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod7>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod7<-as.matrix(test)
proba_prueba_mod7<-h(x_prueba_mod7%*%as.numeric(coef_mod7_Ridge))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod7<-ifelse(proba_prueba_mod7>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod7,"Result_prueba_mod7.csv")

#---- Modelo 8: Reg Ridge (Sin Horas ni Minuos, estandarizado)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod8<-as.matrix(train_norm[,1:38000])

# Corremos un modelo glm.
mod8<-glmnet(x=x_mod8,y=train_norm$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Obtenemos todos los coeficientes del modelo
coef_mod8_Ridge<-coef(mod8)[-1]

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod8<-h(x_mod8%*%as.numeric(coef_mod8_Ridge))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod8<-ifelse(proba_entrena_mod8>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod8>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod8<-as.matrix(test_norm[,1:38000])
proba_prueba_mod8<-h(x_prueba_mod8%*%as.numeric(coef_mod8_Ridge))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod8<-ifelse(proba_prueba_mod8>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod8,"Result_prueba_mod8.csv")

#---- Modelo 9: Reg Ridge (Con Horas y Minuos, estandarizado)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod9<-as.matrix(train_norm[,c(1:38000,38002:ncol(train_norm))])

# Corremos un modelo glm.
mod9<-glmnet(x=x_mod9,y=train_norm$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             lambda=0.1)

# Obtenemos todos los coeficientes del modelo
coef_mod9_Ridge<-coef(mod9)[-1]

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod9<-h(x_mod9%*%as.numeric(coef_mod9_Ridge))

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod9<-ifelse(proba_entrena_mod9>0.5,1,0)

# Obtenemos la matriz de confusión
table(proba_entrena_mod9>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod9<-as.matrix(test_norm)
proba_prueba_mod9<-h(x_prueba_mod9%*%as.numeric(coef_mod9_Ridge))

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod9<-ifelse(proba_prueba_mod9>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod9,"Result_prueba_mod9.csv")




#---- Modelo 10: Reg Lasso (Sin Horas ni Minuos, sin estandarizar,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod10<-as.matrix(train[,1:38000])

set.seed(12345)
# Corremos un modelo glm.
mod10<-cv.glmnet(x=x_mod10,y=train$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod10<-predict(mod10,newx=x_mod10,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod10<-ifelse(proba_entrena_mod10>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod10<-mod10$cvm[mod10$lambda==mod10$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod10>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod10<-as.matrix(test[,1:38000])
proba_prueba_mod10<-predict(mod10,newx=x_prueba_mod10,s='lambda.min',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod10<-ifelse(proba_prueba_mod10>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod10,"Result_prueba_mod10.csv")

#---- Modelo 11: Reg Lasso (Con Horas y Minuos, sin estandarizar,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
set.seed(12345)
x_mod11<-as.matrix(train[,c(1:38000,38002:ncol(train))])

# Corremos un modelo glm.
mod11<-cv.glmnet(x=x_mod11,y=train$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             parallel=TRUE,
             nfolds=10,
             lambda=exp(seq(-12,2,1)))

plot(mod11)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod11<-predict(mod11,newx=x_mod11,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod11<-ifelse(proba_entrena_mod11>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod11<-mod11$cvm[mod11$lambda==mod11$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod11>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod11<-as.matrix(test)
proba_prueba_mod11<-predict(mod11,newx=x_prueba_mod11,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod11<-ifelse(proba_prueba_mod11>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod11,"Result_prueba_mod11.csv")

#---- Modelo 12: Reg Lasso (Sin Horas ni Minuos, estandarizado,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod12<-as.matrix(train_norm[,1:38000])

set.seed(12345)
# Corremos un modelo glm.
mod12<-cv.glmnet(x=x_mod12,y=train_norm$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod12<-predict(mod12,newx=x_mod12,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod12<-ifelse(proba_entrena_mod12>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod12<-mod12$cvm[mod12$lambda==mod12$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod12>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod12<-as.matrix(test_norm[,1:38000])
proba_prueba_mod12<-predict(mod12,newx=x_prueba_mod12,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod12<-ifelse(proba_prueba_mod12>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod12,"Result_prueba_mod12.csv")

#---- Modelo 13: Reg Lasso (Con Horas y Minuos, estandarizado,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod13<-as.matrix(train_norm[,c(1:38000,38002:ncol(train_norm))])

set.seed(12345)
# Corremos un modelo glm.
mod13<-cv.glmnet(x=x_mod13,y=train_norm$estado,
             alpha=1,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod13<-predict(mod13,newx=x_mod13,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod13<-ifelse(proba_entrena_mod13>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod13<-mod13$cvm[mod13$lambda==mod13$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod13>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod13<-as.matrix(test_norm)
proba_prueba_mod13<-predict(mod13,newx=x_prueba_mod13,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod13<-ifelse(proba_prueba_mod13>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod13,"Result_prueba_mod13.csv")

#---- Modelo 14: Reg Ridge (Sin Horas ni Minuos, sin estandarizar,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod14<-as.matrix(train[,1:38000])

set.seed(12345)
# Corremos un modelo glm.
mod14<-cv.glmnet(x=x_mod14,y=train$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod14<-predict(mod14,newx=x_mod14,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod14<-ifelse(proba_entrena_mod14>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod14<-mod14$cvm[mod14$lambda==mod14$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod14>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod14<-as.matrix(test[,1:38000])
proba_prueba_mod14<-predict(mod14,newx=x_prueba_mod14,s='lambda.1se',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod14<-ifelse(proba_prueba_mod14>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod14,"Result_prueba_mod14.csv")

#---- Modelo 15: Reg Ridge (Con Horas y Minuos, sin estandarizar,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod15<-as.matrix(train[,c(1:38000,38002:ncol(train))])

set.seed(12345)
# Corremos un modelo glm.
mod15<-cv.glmnet(x=x_mod15,y=train$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod15<-predict(mod15,newx=x_mod15,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod15<-ifelse(proba_entrena_mod15>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod15<-mod15$cvm[mod15$lambda==mod15$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod15>0.5,train$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod15<-as.matrix(test)
proba_prueba_mod15<-predict(mod15,newx=x_prueba_mod15,s="lambda.1se",type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod15<-ifelse(proba_prueba_mod15>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod15,"Result_prueba_mod15.csv")

#---- Modelo 16: Reg Ridge (Sin Horas ni Minuos, estandarizado,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.
#       subimos estas primeras probabiliades al concurso sólo para probar
#       que funcionara. 

# Extraemos las enradas
x_mod16<-as.matrix(train_norm[,1:38000])

set.seed(12345)
# Corremos un modelo glm.
mod16<-cv.glmnet(x=x_mod16,y=train_norm$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod16<-predict(mod16,newx=x_mod16,s="lambda.1se",type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod16<-ifelse(proba_entrena_mod16>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod16<-mod16$cvm[mod16$lambda==mod16$lambda.1se]


# Obtenemos la matriz de confusión
table(proba_entrena_mod16>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod16<-as.matrix(test_norm[,1:38000])
proba_prueba_mod16<-predict(mod16,newx=x_prueba_mod16,s='lambda.1se',type='response')

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod16<-ifelse(proba_prueba_mod16>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod16,"Result_prueba_mod16.csv")

#---- Modelo 17: Reg Ridge (Con Horas y Minuos, estandarizado,crossvalidation)----
# Nota: no utilziamos las horas y los minutos.

# Extraemos las enradas
x_mod17<-as.matrix(train_norm[,c(1:38000,38002:ncol(train_norm))])

set.seed(12345)
# Corremos un modelo glm.
mod17<-cv.glmnet(x=x_mod17,y=train_norm$estado,
             alpha=0,
             family='binomial',
             intercept=FALSE,
             nfolds=10,
             nlambda=50)

# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod17<-predict(mod17,newx=x_mod17,s="lambda.1se",type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_norm_mod17<-ifelse(proba_entrena_mod17>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod17<-mod17$cvm[mod17$lambda==mod17$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod17>0.5,train_norm$estado)


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod17<-as.matrix(test_norm)
proba_prueba_mod17<-predict(mod17,newx=x_prueba_mod17,s='lambda.1se',type='response')

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod17<-ifelse(proba_prueba_mod17>0.5,1,0)

# Exportamos los resultados del modelo 3.
write.csv(proba_prueba_mod17,"Result_prueba_mod17.csv")




#---- Modelo 18: Ridge dividiendo train en valiación y entrenamiento. -----



#---- RESULTADOS -----
# Hacemos un data frame de las probabilidades de cada modelo. 
probas_modelos<-data.frame(Prob2<-proba_entrena_mod2,
                           Prob3<-proba_entrena_mod3,
                           Proba4<-proba_entrena_mod4,
                           Proba5<-proba_entrena_mod5,
                           Proba6<-proba_entrena_mod6,
                           Proba7<-proba_entrena_mod7,
                           Proba8<-proba_entrena_mod8,
                           Proba9<-proba_entrena_mod9,
                           Proba10<-proba_entrena_mod10,
                           Proba11<-proba_entrena_mod11,
                           Proba12<-proba_entrena_mod12,
                           Proba13<-proba_entrena_mod13,
                           Proba14<-proba_entrena_mod14,
                           Proba15<-proba_entrena_mod15,
                           Proba16<-proba_entrena_mod16,
                           Proba17<-proba_entrena_mod17)

perdidalog_entrena<-vector("numeric",16)

# Calculamos la perdida logaritmica de entrenamiento.
for(i in 1:16){
  perdidalog_entrena[i]<-devianza(train$estado,probas_modelos[,i])
}

# Calculamos la devianzas de los modelos con cross validation.
perdidalog_validación<-c(rep(NA,length(2:9)),
                         dev_cv_mod10,dev_cv_mod11,dev_cv_mod12,dev_cv_mod13,
                         dev_cv_mod14,dev_cv_mod15,dev_cv_mod16,dev_cv_mod17)
ResultadosBasicos<-data.frame(Modelo=2:17,Perdida_Entrna=perdidalog_entrena,Perdida_CV=perdidalog_validación)
write.csv(ResultadosBasicos,"ResultadosBasicos_glmnet.csv")

