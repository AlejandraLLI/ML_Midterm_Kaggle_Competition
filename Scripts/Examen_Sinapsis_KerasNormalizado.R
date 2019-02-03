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
entrena2<-train

entrena <- entrena %>%
  mutate(estado=ifelse(estado=='abierta',0,1),
         hour= hour(hora), 
         minutes= minute(hora),
         hora=as.numeric(hora))

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




#---- mod1: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod1<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod1<-train_norm$estado %>% as.vector

x_valid_mod1<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod1<-validation_norm$estado  %>% as.vector

x_test_mod1<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod1<-keras_model_sequential()

# Damos estructura a la red. 
mod1 %>% 
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod1))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))
  
# Definimos la perdida y el porcentaje de correctos
mod1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod1 %>% fit(
  x_train_mod1,y_train_mod1, 
  epochs=noIter, batch_size=nrow(x_train_mod1),
  verbose=1,
  validation_data=list(x_valid_mod1,y_valid_mod1)
)

score_mod1<-mod1 %>% evaluate(x_valid_mod1,y_valid_mod1)
score_mod1

tab_confusion_mod1<-table(mod1 %>% predict_classes(x_valid_mod1),y_valid_mod1)
tab_confusion_mod1

prop.table(tab_confusion_mod1,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod1<-mod1 %>% predict_proba(x_test_mod1,batch_size=nrow(x_test_mod1))

write.csv(probas_prueba_mod1,"mod1_RedesNeuronales.csv")

#---- mod2: 4 capas (100 unidades en ocultas), datos no estandarizados, lambda=1e-4, eta=0.1 ----
x_train_mod2<-train[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod2<-train$estado %>% as.vector

x_valid_mod2<-validation[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod2<-validation$estado  %>% as.vector

x_test_mod2<-test %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod2<-keras_model_sequential()

# Damos estructura a la red. 
mod2 %>% 
  
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod2))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod2 %>% fit(
  x_train_mod2,y_train_mod2, 
  epochs=noIter, batch_size=nrow(x_train_mod2),
  verbose=1,
  validation_data=list(x_valid_mod2,y_valid_mod2)
)

score_mod2<-mod2 %>% evaluate(x_valid_mod2,y_valid_mod2)
score_mod2

tab_confusion_mod2<-table(mod2 %>% predict_classes(x_valid_mod2),y_valid_mod2)
tab_confusion_mod2

prop.table(tab_confusion_mod2,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod2<-mod2 %>% predict_proba(x_test_mod2,batch_size=nrow(x_test_mod2))

write.csv(probas_prueba_mod2,"mod2_RedesNeuronales.csv")

#---- mod3: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-2, eta=0.1 ----
x_train_mod3<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod3<-train_norm$estado %>% as.vector

x_valid_mod3<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod3<-validation_norm$estado  %>% as.vector

x_test_mod3<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod3<-keras_model_sequential()

# Damos estructura a la red. 
mod3 %>% 
  
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-2),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod3))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-2)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-2)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod3 %>% fit(
  x_train_mod3,y_train_mod3, 
  epochs=noIter, batch_size=nrow(x_train_mod3),
  verbose=1,
  validation_data=list(x_valid_mod3,y_valid_mod3)
)

score_mod3<-mod3 %>% evaluate(x_valid_mod3,y_valid_mod3)
score_mod3

tab_confusion_mod3<-table(mod3 %>% predict_classes(x_valid_mod3),y_valid_mod3)
tab_confusion_mod3

prop.table(tab_confusion_mod3,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod3<-mod3 %>% predict_proba(x_test_mod3,batch_size=nrow(x_test_mod3))

write.csv(probas_prueba_mod3,"mod3_RedesNeuronales.csv")


#---- mod4: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.001 ----
x_train_mod4<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod4<-train_norm$estado %>% as.vector

x_valid_mod4<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod4<-validation_norm$estado  %>% as.vector

x_test_mod4<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod4<-keras_model_sequential()

# Damos estructura a la red. 
mod4 %>% 
  
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod4))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod4 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.001),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod4 %>% fit(
  x_train_mod4,y_train_mod4, 
  epochs=noIter, batch_size=nrow(x_train_mod4),
  verbose=1,
  validation_data=list(x_valid_mod4,y_valid_mod4)
)

score_mod4<-mod4 %>% evaluate(x_valid_mod4,y_valid_mod4)
score_mod4

tab_confusion_mod4<-table(mod4 %>% predict_classes(x_valid_mod4),y_valid_mod4)
tab_confusion_mod4

prop.table(tab_confusion_mod4,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod4<-mod4 %>% predict_proba(x_test_mod4,batch_size=nrow(x_test_mod4))

write.csv(probas_prueba_mod4,"mod4_RedesNeuronales.csv")


#---- mod5: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod5<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod5<-train_norm$estado %>% as.vector

x_valid_mod5<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod5<-validation_norm$estado  %>% as.vector

x_test_mod5<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod5<-keras_model_sequential()

# Damos estructura a la red. 
mod5 %>% 
  
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod5))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod5 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod5 %>% fit(
  x_train_mod5,y_train_mod5, 
  epochs=noIter, batch_size=nrow(x_train_mod5),
  verbose=1,
  validation_data=list(x_valid_mod5,y_valid_mod5)
)

score_mod5<-mod5 %>% evaluate(x_valid_mod5,y_valid_mod5)
score_mod5

tab_confusion_mod5<-table(mod5 %>% predict_classes(x_valid_mod5),y_valid_mod5)
tab_confusion_mod5

prop.table(tab_confusion_mod5,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod5<-mod5 %>% predict_proba(x_test_mod5,batch_size=nrow(x_test_mod5))

write.csv(probas_prueba_mod5,"mod5_RedesNeuronales.csv")


#---- mod6: 4 capas (100 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=1 ----
x_train_mod6<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod6<-train_norm$estado %>% as.vector

x_valid_mod6<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod6<-validation_norm$estado  %>% as.vector

x_test_mod6<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod6<-keras_model_sequential()

# Damos estructura a la red. 
mod6 %>% 
  
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod6))%>%
  layer_dense(units=100,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod6 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=1),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod6 %>% fit(
  x_train_mod6,y_train_mod6, 
  epochs=noIter, batch_size=nrow(x_train_mod6),
  verbose=1,
  validation_data=list(x_valid_mod6,y_valid_mod6)
)

score_mod6<-mod6 %>% evaluate(x_valid_mod6,y_valid_mod6)
score_mod6

tab_confusion_mod6<-table(mod6 %>% predict_classes(x_valid_mod6),y_valid_mod6)
tab_confusion_mod6

prop.table(tab_confusion_mod6,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod6<-mod6 %>% predict_proba(x_test_mod6,batch_size=nrow(x_test_mod6))

write.csv(probas_prueba_mod6,"mod6_RedesNeuronales.csv")



#---- mod7: 4 capas (500 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod7<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod7<-train_norm$estado %>% as.vector

x_valid_mod7<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod7<-validation_norm$estado  %>% as.vector

x_test_mod7<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod7<-keras_model_sequential()

# Damos estructura a la red. 
mod7 %>% 
  
  layer_dense(units=500,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod7))%>%
  layer_dense(units=500,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod7 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod7 %>% fit(
  x_train_mod7,y_train_mod7, 
  epochs=noIter, batch_size=nrow(x_train_mod7),
  verbose=1,
  validation_data=list(x_valid_mod7,y_valid_mod7)
)

score_mod7<-mod7 %>% evaluate(x_valid_mod7,y_valid_mod7)
score_mod7

tab_confusion_mod7<-table(mod7 %>% predict_classes(x_valid_mod7),y_valid_mod7)
tab_confusion_mod7

prop.table(tab_confusion_mod7,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod7<-mod7 %>% predict_proba(x_test_mod7,batch_size=nrow(x_test_mod7))

write.csv(probas_prueba_mod7,"mod7_RedesNeuronales.csv")


#---- mod8: 4 capas (50 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod8<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod8<-train_norm$estado %>% as.vector

x_valid_mod8<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod8<-validation_norm$estado  %>% as.vector

x_test_mod8<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod8<-keras_model_sequential()

# Damos estructura a la red. 
mod8 %>% 
  
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod8))%>%
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod8 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod8 %>% fit(
  x_train_mod8,y_train_mod8, 
  epochs=noIter, batch_size=nrow(x_train_mod8),
  verbose=1,
  validation_data=list(x_valid_mod8,y_valid_mod8)
)

score_mod8<-mod8 %>% evaluate(x_valid_mod8,y_valid_mod8)
score_mod8

tab_confusion_mod8<-table(mod8 %>% predict_classes(x_valid_mod8),y_valid_mod8)
tab_confusion_mod8

prop.table(tab_confusion_mod8,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod8<-mod8 %>% predict_proba(x_test_mod8,batch_size=nrow(x_test_mod8))

write.csv(probas_prueba_mod8,"mod8_RedesNeuronales.csv")


#---- mod9: 4 capas (10 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod9<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod9<-train_norm$estado %>% as.vector

x_valid_mod9<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod9<-validation_norm$estado  %>% as.vector

x_test_mod9<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod9<-keras_model_sequential()

# Damos estructura a la red. 
mod9 %>% 
  
  layer_dense(units=10,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod9))%>%
  layer_dense(units=10,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod9 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod9 %>% fit(
  x_train_mod9,y_train_mod9, 
  epochs=noIter, batch_size=nrow(x_train_mod9),
  verbose=1,
  validation_data=list(x_valid_mod9,y_valid_mod9)
)

score_mod9<-mod9 %>% evaluate(x_valid_mod9,y_valid_mod9)
score_mod9

tab_confusion_mod9<-table(mod9 %>% predict_classes(x_valid_mod9),y_valid_mod9)
tab_confusion_mod9

prop.table(tab_confusion_mod9,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod9<-mod9 %>% predict_proba(x_test_mod9,batch_size=nrow(x_test_mod9))

write.csv(probas_prueba_mod9,"mod9_RedesNeuronales.csv")



#---- mod10: 4 capas (50 unidades en ocultas), datos estandarizados, lambda=1e-2, eta=0.5 ----
x_train_mod10<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod10<-train_norm$estado %>% as.vector

x_valid_mod10<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod10<-validation_norm$estado  %>% as.vector

x_test_mod10<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod10<-keras_model_sequential()

# Damos estructura a la red. 
mod10 %>% 
  
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-2),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod10))%>%
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-2)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-2)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod10 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod10 %>% fit(
  x_train_mod10,y_train_mod10, 
  epochs=noIter, batch_size=nrow(x_train_mod10),
  verbose=1,
  validation_data=list(x_valid_mod10,y_valid_mod10)
)

score_mod10<-mod10 %>% evaluate(x_valid_mod10,y_valid_mod10)
score_mod10

tab_confusion_mod10<-table(mod10 %>% predict_classes(x_valid_mod10),y_valid_mod10)
tab_confusion_mod10

prop.table(tab_confusion_mod10,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod10<-mod10 %>% predict_proba(x_test_mod10,batch_size=nrow(x_test_mod10))

write.csv(probas_prueba_mod10,"mod10_RedesNeuronales.csv")


#---- mod11: 4 capas (50 unidades en ocultas), datos estandarizados, lambda=1e-6, eta=0.5 ----
x_train_mod11<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod11<-train_norm$estado %>% as.vector

x_valid_mod11<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod11<-validation_norm$estado  %>% as.vector

x_test_mod11<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod11<-keras_model_sequential()

# Damos estructura a la red. 
mod11 %>% 
  
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-6),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod11))%>%
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2((l=1e-6)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-6)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod11 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod11 %>% fit(
  x_train_mod11,y_train_mod11, 
  epochs=noIter, batch_size=nrow(x_train_mod11),
  verbose=1,
  validation_data=list(x_valid_mod11,y_valid_mod11)
)

score_mod11<-mod11 %>% evaluate(x_valid_mod11,y_valid_mod11)
score_mod11

tab_confusion_mod11<-table(mod11 %>% predict_classes(x_valid_mod11),y_valid_mod11)
tab_confusion_mod11

prop.table(tab_confusion_mod11,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod11<-mod11 %>% predict_proba(x_test_mod11,batch_size=nrow(x_test_mod11))

write.csv(probas_prueba_mod11,"mod11_RedesNeuronales.csv")



#---- mod12: 3 capas (50 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod12<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod12<-train_norm$estado %>% as.vector

x_valid_mod12<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod12<-validation_norm$estado  %>% as.vector

x_test_mod12<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod12<-keras_model_sequential()

# Damos estructura a la red. 
mod12 %>% 
  
  layer_dense(units=50,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod12))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod12 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod12 %>% fit(
  x_train_mod12,y_train_mod12, 
  epochs=noIter, batch_size=nrow(x_train_mod12),
  verbose=1,
  validation_data=list(x_valid_mod12,y_valid_mod12)
)

score_mod12<-mod12 %>% evaluate(x_valid_mod12,y_valid_mod12)
score_mod12

tab_confusion_mod12<-table(mod12 %>% predict_classes(x_valid_mod12),y_valid_mod12)
tab_confusion_mod12

prop.table(tab_confusion_mod12,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod12<-mod12 %>% predict_proba(x_test_mod12,batch_size=nrow(x_test_mod12))

write.csv(probas_prueba_mod12,"mod12_RedesNeuronales.csv")



#---- mod13: 3 capas (25 unidades en ocultas), datos estandarizados, lambda=1e-4, eta=0.5 ----
x_train_mod13<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_mod13<-train_norm$estado %>% as.vector

x_valid_mod13<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_mod13<-validation_norm$estado  %>% as.vector

x_test_mod13<-test_norm %>% as.matrix

# Fijamos la semilla.
set.seed(12345)
# set.seed(34323)

# Creamos la red
mod13<-keras_model_sequential()

# Damos estructura a la red. 
mod13 %>% 
  
  layer_dense(units=25,activation='sigmoid',
              kernel_regularizer = regularizer_l2(l=1e-4),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5),
              input_shape=ncol(x_train_mod13))%>%
  layer_dense(units=1,activation='sigmoid',
              kernel_regularizer=regularizer_l2((l=1e-4)),
              kernel_initializer=initializer_random_uniform(minval=-0.5,maxval=0.5))

# Definimos la perdida y el porcentaje de correctos
mod13 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr=0.5),
  metrics = c('accuracy','binary_crossentropy'))

# Iteramos (100 veces) con descenso en gradiente y monitoreamos el error de validación
noIter=500

# Iniciamos las iteraciones. 
iteraciones<-mod13 %>% fit(
  x_train_mod13,y_train_mod13, 
  epochs=noIter, batch_size=nrow(x_train_mod13),
  verbose=1,
  validation_data=list(x_valid_mod13,y_valid_mod13)
)

score_mod13<-mod13 %>% evaluate(x_valid_mod13,y_valid_mod13)
score_mod13

tab_confusion_mod13<-table(mod13 %>% predict_classes(x_valid_mod13),y_valid_mod13)
tab_confusion_mod13

prop.table(tab_confusion_mod13,2)

df_iteraciones<-as.data.frame(iteraciones)

ggplot(df_iteraciones,aes(x=epoch,y=value,colour=data,group=data))+
  geom_line()+geom_point()+facet_wrap(~metric,ncol=1,scales='free')

probas_prueba_mod13<-mod13 %>% predict_proba(x_test_mod13,batch_size=nrow(x_test_mod13))

write.csv(probas_prueba_mod13,"mod13_RedesNeuronales.csv")







#---- Resultados ----
SCORES<-data.frame(Modelo=1:13,
                   Loss=c(score_mod1$loss,score_mod2$loss,score_mod3$loss,score_mod4$loss,
                          score_mod5$loss,score_mod6$loss,score_mod7$loss,score_mod8$loss,
                          score_mod9$loss,score_mod10$loss,score_mod11$loss,score_mod12$loss,score_mod13$loss),
                   Accuracy=c(score_mod1$acc,score_mod2$acc,score_mod3$acc,score_mod4$acc,
                          score_mod5$acc,score_mod6$acc,score_mod7$acc,score_mod8$acc,
                          score_mod9$acc,score_mod10$acc,score_mod11$acc,score_mod12$acc,score_mod13$acc),
                   Binary_Crossentropy=c(score_mod1$binary_crossentropy,score_mod2$binary_crossentropy,
                                         score_mod3$binary_crossentropy,score_mod4$binary_crossentropy,
                                         score_mod5$binary_crossentropy,score_mod6$binary_crossentropy,
                                         score_mod7$binary_crossentropy,score_mod8$binary_crossentropy,
                                         score_mod9$binary_crossentropy,score_mod10$binary_crossentropy,
                                         score_mod11$binary_crossentropy,score_mod12$binary_crossentropy,
                                         score_mod13$binary_crossentropy))

SCORES<-SCORES%>%arrange(Binary_Crossentropy,Accuracy,Loss)

write.csv(SCORES,'Resultados_RedesNeuronales_AMano.csv')

#---- Búsqueda de Hiperparámetros (3 capas)----
x_train_hiper<-train_norm[,c(1:38000,38002:ncol(train))] %>% as.matrix
y_train_hiper<-train_norm$estado %>% as.vector

x_valid_hiper<-validation_norm[,c(1:38000,38002:ncol(validation))]  %>% as.matrix
y_valid_hiper<-validation_norm$estado  %>% as.vector



# Definimos el espacio de los hiperparámetros
hiperparams<-expand.grid(lambda=c(1e-6,1e-4,1e-2),
                         n_capa=c(25,50,100,500),
                         lr=c(0.01,0.1,0.5),
                         init_pesos=c(0.5),
                         stringsAsFactors = FALSE)

#Asignamos numero de corrida a cada conjunto de hiperparametros
hiperparams$corrida<-1:nrow(hiperparams)

# Creamos la función para correr cada modelo de 3 capas. 
correr_modelo_3capas<-function(params,x_train,y_train,x_valid,y_valid){
  
  # Creamos el modelo
  modelo_tc<-keras_model_sequential()
  
  # Seleccionamos el peso inicial. 
  u<-params[['init_pesos']]
  
  # damos estructura al modelo
  modelo_tc %>%
    
    layer_dense(units=params[['n_capa']],
                activation='sigmoid',
                kernel_regularizer = regularizer_l2(l=params[['lambda']]),
                kernel_initializer = initializer_random_uniform(minval=-u,maxval=u),
                input_shape=ncol(x_train)) %>%
    layer_dense(units=1,
                activation='sigmoid',
                kernel_regularizer = regularizer_l2(l=params[['lambda']]),
                kernel_initializer = initializer_random_uniform(minval=-u,maxval=u))
  
  # Definimos perdida y metricas
  modelo_tc%>% compile(loss="binary_crossentropy",
                       optimizer=optimizer_sgd(lr=params[['lr']]),
                       metrics=c('accuracy','binary_crossentropy')
                       )
  
  # Definimos las iteraciones
  iteraciones<-modelo_tc %>%fit(x_train,y_train,
                                epochs=params[['n_iter']],
                                batch_size=nrow(x_train),
                                verbose=0)
  
  # Obtenemos resultados
  score<-modelo_tc %>% evaluate(x_valid,y_valid)
  
  print(score)
  
  score
    
}

# Fijamos la semilla
set.seed(12345)

nombres<-names(hiperparams)


#corremos los modelos
# if(!usar_cache){

  res<-lapply(1:nrow(hiperparams)
              ,function(i){
                params<-as.vector(hiperparams[i,])
                salida<-correr_modelo_3capas(params,x_train_hiper,y_train_hiper,x_valid_hiper,y_valid_hiper)
                salida
                }
    )
  hiperparams$binary_crossentropy<-sapply(res,function(item){item$binary_crossentropy})
  hiperparams$loss<-sapply(res,function(item){item$loss})
  hiperparams$acc<-sapply(res,function(item){item$acc})
#   saveRDS(hiperparams,file='./cache_obj/examen-grid.rds')
# }else{
#   hiperparams<-readRDS(file='./cache_obj/examen-grid.rds')
# }

  hiperparams$binary_crossentropy<-sapply(res,function(item){item$binary_crossentropy})
  hiperparams$loss<-sapply(res,function(item){item$loss})
  hiperparams$acc<-sapply(res,function(item){item$acc})

# ordenamos del mejor al peor modelo
arrange(hiperparams, binary_crossentropy)