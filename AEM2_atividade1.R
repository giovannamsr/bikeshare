

#Atividade 1

# utilizaremos a base de dados Bikeshare do pacote ISLR2 feito pela UCI
# Faremos a projeção de alugueis de bike em uma granularidade de hora
# Não será usado metodologia de time-series, mas de regressão usual.  

#Grupo:
# Arnaldo, Caetano, Daniel, Giovanna



# Libraries ---------------------------------------------------------------

library(tidyverse)
library(ISLR2)
library(rsample)
library(skimr)
library(tidymodels)
library(recipes)
library(vip)
library(doParallel)


# input de dados ----------------------------------------------------------

dados<- Bikeshare
dados %>% skim()


# ajustes básicos de dados --------------------------------------------------------


#retirar variavel 'casual' e 'registered' para evitar dataleak
df<- dados %>% dplyr::select(-c(casual,registered))
skim(df)


# transformando em fatores variáveis categoricas que estao como numericas
df <- df %>% 
  mutate(season=as.factor(season),
         holiday=as.factor(holiday),
         weekday=as.factor(weekday),
         workingday=as.factor(workingday))
skim(df)


## analisando as variaveis categoricas e realizando ajustes se necessario

categorical_variables <- df %>% 
  select_if(is.factor) %>% 
  names()

for (i in categorical_variables) {
  cat("Variable:", i, "\n")
  print(table(df[[i]]))
  cat("\n")
}

# Variavel Weathersit tem uma categoria com apenas 1 dado nela, iremos retirar pois não será possível
# Realiazar a regressão dessa forma

df <- df %>% filter(weathersit != 'heavy rain/snow')



# Training - test split ---------------------------------------------------

set.seed(321)

split <- initial_split(df, prop = 0.8)
treinamento <- training(split)
teste <- testing(split)


# Tibble de resultados ----------------------------------------------------


fitted <- tibble(
  .pred = numeric(0),
  observado = numeric(0),
  modelo = character(0))



# Modelo1 - formato usual -------------------------------------------------





# Modelo 1 - formato tidymodels -------------------------------------------







# XGBoost - formato antigo-------------------------------------------------



# XGBoost - formato tidymodels --------------------------------------------





# Resultados e comentários finais -----------------------------------------






