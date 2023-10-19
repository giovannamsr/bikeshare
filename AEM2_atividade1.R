

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



# Analise da variavel objetivo --------------------------------------------

dados %>% 
  ggplot(aes(x=bikers)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Histogram of bikes rentals per hour", x = "bikers", y = "Frequency")+
  theme_bw()

# dado o skew da variável, será interessante transformarmos a variavel para Log

dados %>% 
  ggplot(aes(x=hr,y=bikers,color=season,group=day))+
  geom_line()



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


#receita
(receita <- recipe(bikers ~ ., data = treinamento) %>% 
   step_log(bikers) %>%  #transforma em ln a variavel objetivo
   step_normalize(all_numeric(), -all_outcomes()) %>%  #normalizando as variaveis
   step_dummy(all_nominal(), -all_outcomes())) # dummy enconding


#prep receita
(receita_prep <- prep(receita)) # prepara a receita definida acima - pre processamento do recipe


# bake na base de teste e treino
treinamento_proc <- bake(receita_prep, new_data = NULL) # obtem os dados de treinamento processados
teste_proc <- bake(receita_prep, new_data = teste) # obtem os dados de teste processados

# checando os dados processados
skim(treinamento_proc)
skim(teste_proc)


# Modelo

boost <- boost_tree( trees = tune(), learn_rate = tune(), mtry = tune(), 
                     tree_depth = tune(), min_n = tune(), sample_size = tune()) %>% 
  set_engine('xgboost') %>% 
  set_mode('regression')

# cv
set.seed(123)
cv_split <- vfold_cv(treinamento, v=10)


# otimização de hiperparametro
doParallel::registerDoParallel(makeCluster(8))  #alterar para o numero de cores da sua maquina
set.seed(123)
boost_grid <- tune_grid(boost,
                        receita,
                        resamples = cv_split,
                        grid = 50,
                        metrics = metric_set(rmse))

autoplot(boost_grid) # plota os resultados do grid search

boost_grid %>% 
  collect_metrics()   # visualiza o tibble de resultados

(best_xgb <- boost_grid %>% 
    select_best("rmse")) # salva o melhor conjunto de parametros


#finaliza o modelo
boost_fit <- finalize_model(boost, parameters= best_xgb) %>% 
  fit(bikers ~ ., teste_proc )

# tibble de predições
fitted_xgb <- boost_fit %>% 
  predict(new_data = teste_proc) %>% 
  mutate(observado = teste_proc$bikers,
         modelo = 'xgboost tune - tidymodels')


# visualizando variáveis mais importantes
vip(boost_fit)


#plot predito vs realizado em base LN
fitted_xgb %>% 
  ggplot(aes(x=observado, y=.pred)) +
  geom_point(size=1, col= 'blue') +
  geom_abline(intercept = 0, slope = 1,color = "red", linetype = "dashed", linewidth = 0.7) +
  labs(y = 'predito xgboost', x = 'Observado') +
  theme_bw()

#plot predito vs realizado na base original
fitted_xgb %>% 
  ggplot(aes(x=exp(observado),y= exp(.pred))) +
  geom_point(size=1, col= 'blue') +
  geom_abline(intercept = 0, slope = 1,color = "red", linetype = "dashed", linewidth = 0.7) +
  labs(y = 'predito xgboost', x = 'Observado') +
  theme_bw()

# adicionando ao tibble de resultados
fitted <-  fitted %>% 
  bind_rows(fitted_xgb)


# Resultados e comentários finais -----------------------------------------


fitted %>% 
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) %>% 
  filter(.metric=='rmse')







