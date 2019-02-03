#---- Cargamos paquetes----
# install.packages("tidyverse")
# install.packages("data.table")
# install.packages("glmnet")
library(tidyverse)
library(data.table)
library(glmnet)
library(keras)
library(readr)
#---- Función devianza ----
devianza<-function(estado,probs){
  
  # No de observaciones
  n<-length(estado)
  
  # Pérdida logística
  mean(-(estado*log(probs)+(1-estado)*log(1-probs)))
  
}

devianza_reg<-function(estado,probs,lambda,coef){
  
  # No de observaciones
  n<-length(estado)
  
  # Pérdida logística
  mean(-(estado*log(probs)+(1-estado)*log(1-probs))+lambda*sum(coef^2))
  
}
#---- Leemos los datos de entrenamiento ----
entrena<-read_csv('train.csv')
entrena2<-entrena

entrena <- entrena %>%
  mutate(estado=ifelse(estado=='abierta',0,1),
         hour= hour(hora), 
         minutes= minute(hora),
         hora=as.numeric(hora))

entrena<-entrena %>%
  mutate(claro=ifelse(hour<6,-1,ifelse(hour<10,0,ifelse(hour<17,1,ifelse(hour<19,0,-1)))))%>%
  select(-c(hora,hour,minutes))


#---- Dividimos en muestra de entrenamiento y de validación ----

# Seleccionamos el porcentaje de datos que formarán parte de los datos de entrenamiento y de validación.
fractionTraining <- 0.7
fractionValidation <- 0.3

# Obtenemos el tamaño de muestra de las fracciones anteriores.
sampleSizeTraining <- floor(fractionTraining * nrow(entrena))
sampleSizeValidation <- floor(fractionValidation * nrow(entrena))


#Creamos los índices aleatoriamente.
set.seed(584852)
indicesTraining <- sort(sample(seq_len(nrow(entrena)), size=sampleSizeTraining))

#Creamos los dataframes de entrenamiento y validación a partir de lo anterior.
train   <- entrena[indicesTraining, ]
validation <- entrena[-indicesTraining, ]

#---- Normalizamos los datos de entrenamiento ----
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
train<-train %>% select(-id)
cols<-match(colnames(train),colnames(train_norm))
train_norm<-train_norm[,cols]



#---- Normalizamos los datos de validación----
validation$id<-1:nrow(validation)

validation_MeanSD<-validation %>% 
  gather(Variable, Valor, -estado) %>% # Apilamos
  group_by(Variable) %>% # Agrupamos por variable
  summarise(media=mean(Valor), de=sd(Valor)) # calcula media y desviación.

# Se normalizan los datos de entranmiento.
validation_norm<-normalizar_train(validation,train_MeanSD)

# Se reordenan las columnas
validation<-validation%>%select(-id)
cols<-match(colnames(validation),colnames(validation_norm))
validation_norm<-validation_norm[,cols]


#---- Leemos los datos de prueba ----
test<-read_csv('test.csv')
test2<-test

test <- test %>%
  mutate(hour= hour(hora), 
         minutes= minute(hora),
         hora=as.numeric(hora))

test<-test%>%
  mutate(claro=ifelse(hour<6,-1,ifelse(hour<10,0,ifelse(hour<17,1,ifelse(hour<19,0,-1)))))%>%
  select(-c(hora,hour,minutes))


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





#---- mod14: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod14<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod14<-train_norm$estado %>% as.vector

x_valid_mod14<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod14<-validation_norm$estado  %>% as.vector

x_test_mod14<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod14<-keras_model_sequential()

# Damos estructura a la red. 
mod14 %>% 
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod14))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod14 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod14 %>% fit(
  x_train_mod14,y_train_mod14, 
  epochs=noIter, batch_size=nrow(x_train_mod14),
  verbose=1,
  validation_data=list(x_valid_mod14,y_valid_mod14)
)

score_mod14<-mod14 %>% evaluate(x_valid_mod14,y_valid_mod14)
score_mod14

tab_confusion_mod14<-table(mod14 %>% predict_classes(x_valid_mod14),y_valid_mod14)
tab_confusion_mod14

prop.table(tab_confusion_mod14,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod14<-mod14 %>% predict_proba(x_test_mod14,batch_size=nrow(x_test_mod14))

write.csv(probas_prueba_mod14,"mod14_RedesNeuronales.csv")


#---- mod15: 4 capas (250 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod15<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod15<-train_norm$estado %>% as.vector

x_valid_mod15<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod15<-validation_norm$estado  %>% as.vector

x_test_mod15<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod15<-keras_model_sequential()

# Damos estructura a la red. 
mod15 %>% 
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod15))%>%
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod15 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod15 %>% fit(
  x_train_mod15,y_train_mod15, 
  epochs=noIter, batch_size=nrow(x_train_mod15),
  verbose=1,
  validation_data=list(x_valid_mod15,y_valid_mod15)
)

score_mod15<-mod15 %>% evaluate(x_valid_mod15,y_valid_mod15)
score_mod15

tab_confusion_mod15<-table(mod15 %>% predict_classes(x_valid_mod15),y_valid_mod15)
tab_confusion_mod15

prop.table(tab_confusion_mod15,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod15<-mod15 %>% predict_proba(x_test_mod15,batch_size=nrow(x_test_mod15))

write.csv(probas_prueba_mod15,"mod15_RedesNeuronales.csv")


#---- mod16: 3 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod16<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod16<-train_norm$estado %>% as.vector

x_valid_mod16<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod16<-validation_norm$estado  %>% as.vector

x_test_mod16<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod16<-keras_model_sequential()

# Damos estructura a la red. 
mod16 %>% 
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod16))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod16 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod16 %>% fit(
  x_train_mod16,y_train_mod16, 
  epochs=noIter, batch_size=nrow(x_train_mod16),
  verbose=1,
  validation_data=list(x_valid_mod16,y_valid_mod16)
)

score_mod16<-mod16 %>% evaluate(x_valid_mod16,y_valid_mod16)
score_mod16

tab_confusion_mod16<-table(mod16 %>% predict_classes(x_valid_mod16),y_valid_mod16)
tab_confusion_mod16

prop.table(tab_confusion_mod16,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod16<-mod16 %>% predict_proba(x_test_mod16,batch_size=nrow(x_test_mod16))

write.csv(probas_prueba_mod16,"mod16_RedesNeuronales.csv")


#---- mod17: 3 capas (250 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod17<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod17<-train_norm$estado %>% as.vector

x_valid_mod17<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod17<-validation_norm$estado  %>% as.vector

x_test_mod17<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod17<-keras_model_sequential()

# Damos estructura a la red. 
mod17 %>% 
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod17))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod17 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod17 %>% fit(
  x_train_mod17,y_train_mod17, 
  epochs=noIter, batch_size=nrow(x_train_mod17),
  verbose=1,
  validation_data=list(x_valid_mod17,y_valid_mod17)
)

score_mod17<-mod17 %>% evaluate(x_valid_mod17,y_valid_mod17)
score_mod17

tab_confusion_mod17<-table(mod17 %>% predict_classes(x_valid_mod17),y_valid_mod17)
tab_confusion_mod17

prop.table(tab_confusion_mod17,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod17<-mod17 %>% predict_proba(x_test_mod17,batch_size=nrow(x_test_mod17))

write.csv(probas_prueba_mod17,"mod17_RedesNeuronales.csv")


#---- mod18: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.01 ----
x_train_mod18<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod18<-train_norm$estado %>% as.vector

x_valid_mod18<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod18<-validation_norm$estado  %>% as.vector

x_test_mod18<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod18<-keras_model_sequential()

# Damos estructura a la red. 
mod18 %>% 
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod18))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod18 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.01),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod18 %>% fit(
  x_train_mod18,y_train_mod18, 
  epochs=noIter, batch_size=nrow(x_train_mod18),
  verbose=1,
  validation_data=list(x_valid_mod18,y_valid_mod18)
)

score_mod18<-mod18 %>% evaluate(x_valid_mod18,y_valid_mod18)
score_mod18

tab_confusion_mod18<-table(mod18 %>% predict_classes(x_valid_mod18),y_valid_mod18)
tab_confusion_mod18

prop.table(tab_confusion_mod18,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod18<-mod18 %>% predict_proba(x_test_mod18,batch_size=nrow(x_test_mod18))

write.csv(probas_prueba_mod18,"mod18_RedesNeuronales.csv")


#---- mod19: 4 capas (250 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.01 ----
x_train_mod19<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod19<-train_norm$estado %>% as.vector

x_valid_mod19<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod19<-validation_norm$estado  %>% as.vector

x_test_mod19<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod19<-keras_model_sequential()

# Damos estructura a la red. 
mod19 %>% 
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod19))%>%
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod19 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.01),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod19 %>% fit(
  x_train_mod19,y_train_mod19, 
  epochs=noIter, batch_size=nrow(x_train_mod19),
  verbose=1,
  validation_data=list(x_valid_mod19,y_valid_mod19)
)

score_mod19<-mod19 %>% evaluate(x_valid_mod19,y_valid_mod19)
score_mod19

tab_confusion_mod19<-table(mod19 %>% predict_classes(x_valid_mod19),y_valid_mod19)
tab_confusion_mod19

prop.table(tab_confusion_mod19,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod19<-mod19 %>% predict_proba(x_test_mod19,batch_size=nrow(x_test_mod19))

write.csv(probas_prueba_mod19,"mod19_RedesNeuronales.csv")


#---- mod20: 3 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.01 ----
x_train_mod20<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod20<-train_norm$estado %>% as.vector

x_valid_mod20<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod20<-validation_norm$estado  %>% as.vector

x_test_mod20<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod20<-keras_model_sequential()

# Damos estructura a la red. 
mod20 %>% 
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod20))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod20 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.01),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod20 %>% fit(
  x_train_mod20,y_train_mod20, 
  epochs=noIter, batch_size=nrow(x_train_mod20),
  verbose=1,
  validation_data=list(x_valid_mod20,y_valid_mod20)
)

score_mod20<-mod20 %>% evaluate(x_valid_mod20,y_valid_mod20)
score_mod20

tab_confusion_mod20<-table(mod20 %>% predict_classes(x_valid_mod20),y_valid_mod20)
tab_confusion_mod20

prop.table(tab_confusion_mod20,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod20<-mod20 %>% predict_proba(x_test_mod20,batch_size=nrow(x_test_mod20))

write.csv(probas_prueba_mod20,"mod20_RedesNeuronales.csv")


#---- mod21: 3 capas (250 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.01 ----
x_train_mod21<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod21<-train_norm$estado %>% as.vector

x_valid_mod21<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod21<-validation_norm$estado  %>% as.vector

x_test_mod21<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod21<-keras_model_sequential()

# Damos estructura a la red. 
mod21 %>% 
  layer_dense(units=250,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod21))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod21 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.01),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod21 %>% fit(
  x_train_mod21,y_train_mod21, 
  epochs=noIter, batch_size=nrow(x_train_mod21),
  verbose=1,
  validation_data=list(x_valid_mod21,y_valid_mod21)
)

score_mod21<-mod21 %>% evaluate(x_valid_mod21,y_valid_mod21)
score_mod21

tab_confusion_mod21<-table(mod21 %>% predict_classes(x_valid_mod21),y_valid_mod21)
tab_confusion_mod21

prop.table(tab_confusion_mod21,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod21<-mod21 %>% predict_proba(x_test_mod21,batch_size=nrow(x_test_mod21))

write.csv(probas_prueba_mod21,"mod21_RedesNeuronales.csv")



#---- Resultados ----
SCORES<-data.frame(Modelo=14:21,
                   Loss=c(score_mod14$loss,score_mod15$loss,score_mod16$loss,score_mod17$loss,
                          score_mod18$loss,score_mod19$loss,score_mod20$loss,score_mod21$loss),
                   Accuracy=c(score_mod14$acc,score_mod15$acc,score_mod16$acc,score_mod17$acc,
                              score_mod18$acc,score_mod19$acc,score_mod20$acc,score_mod21$acc),
                   Binary_Crossentropy=c(score_mod14$binary_crossentropy,score_mod15$binary_crossentropy,
                                         score_mod16$binary_crossentropy,score_mod17$binary_crossentropy,
                                         score_mod18$binary_crossentropy,score_mod19$binary_crossentropy,
                                         score_mod20$binary_crossentropy,score_mod21$binary_crossentropy))

SCORES<-SCORES%>%arrange(Binary_Crossentropy,Accuracy,Loss)

write.csv(SCORES,'Resultados_RedesNeuronales_AMano.csv')




# Creamos la función logística
h<-function(x){
  1/(1+exp(-x))
}

devianza<-function(estado,probs){
  
  # No de observaciones
  n<-length(estado)
  
  # Pérdida logística
  mean(-(estado*log(probs)+(1-estado)*log(1-probs)))
  
}

devianza_reg<-function(estado,probs,lambda,coef){
  
  # No de observaciones
  n<-length(estado)
  
  # Pérdida logística
  mean(-(estado*log(probs)+(1-estado)*log(1-probs))+lambda*sum(coef^2))
  
}

#---- Modelo 25: Reg Lasso (usando claro, con estandarizar,crossvalidation)----

# Extraemos las enradas
x_mod25<-as.matrix(rbind(train_norm[,c(1:38000,38002)],validation_norm[,c(1:38000,38002)]))

set.seed(12345)
# Corremos un modelo glm.
mod25<-cv.glmnet(x=x_mod25,y=c(train$estado,validation$estado),
                 alpha=1,
                 family='binomial',
                 intercept=FALSE,
                 nfolds=10,
                 lambda=exp(seq(-12,2,1)))


# Error Entrenamiento y Clasificación Entrenamiento

# Obtenemos las probabilidades de clase
proba_entrena_mod25<-predict(mod25,newx=x_mod25,s='lambda.1se',type="response")

# Obtenemos la clasificación de entrenamiento
estado_estim_train_mod25<-ifelse(proba_entrena_mod25>0.5,1,0)

# Obtenemos la devianza del cv. 
dev_cv_mod25<-mod25$cvm[mod25$lambda==mod25$lambda.1se]

# Obtenemos la matriz de confusión
table(proba_entrena_mod25>0.5,c(train$estado,validation$estado))


# Error Prueba y Clasificador Prueba

# Obtenemos las probabilidades de clase para el conjunto de prueba
x_prueba_mod25<-as.matrix(test[,1:38001])
proba_prueba_mod25<-predict(mod25,newx=x_prueba_mod25,s='lambda.min',type="response")

# Obtenemos la clasificación de prueba
estado_estim_prueba_mod25<-ifelse(proba_prueba_mod25>0.5,1,0)

# Exportamos los resultados del modelo 2.
write.csv(proba_prueba_mod25,"Result_prueba_mod25.csv")

