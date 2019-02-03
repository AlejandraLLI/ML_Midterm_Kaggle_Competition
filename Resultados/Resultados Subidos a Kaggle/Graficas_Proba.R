library(tidyverse)
library(ggplot2)

probas<-read.csv("E:/ITAM Maestría/Otoño 2017/Aprendizaje de Máquina/ExamenParcial/Resultados/Resultados Subidos a Kaggle/TablaProbabilidades.csv",head=TRUE)

factores<-colnames(probas)[2:ncol(probas)]

probas<-probas%>%
  gather(Modelo,Proba,Mod_1:Mod_13)%>%
  mutate(Modelo=as.factor(Modelo))%>%
  mutate(Modelo=factor(Modelo,levels=factores))%>%
  arrange(Modelo)
  
  
  

# proba1<-probas%>%filter(Modelo=="Result_Prueba_mod10.csv")
ggplot(probas,aes(x=Id,y=Proba,group=Modelo,colour=Modelo))+
  facet_wrap(~Modelo,nrow=5)+
  geom_point()
