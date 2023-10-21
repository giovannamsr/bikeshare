

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
library(GGally)
library(factoextra)
library(ggrepel)
library(plotly)
library(cluster)


# input de dados ----------------------------------------------------------

dados <- Bikeshare

dados %>% skim() # não há dados faltantes, bikers possui distribuição exponencial # nolint
dados %>% glimpse() # passar pra fator: season, holiday, weekday, workingday



# Analise da variavel objetivo --------------------------------------------

dados %>%
  ggplot(aes(x = bikers)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Histogram of bikes rentals per hour", x = "bikers", y = "Frequency") + # nolint
  theme_bw()

#Dado o skew da variável, será interessante transformarmos a variavel para Log
#Como  vamos fazer modelos baseados em arvore não vamos fazer log de bikers

dados %>%
  ggplot(aes(x = hr, y = bikers, color = season, group = day)) +
  geom_line()

# ajustes básicos de dados -------------------


#retirar variavel 'casual' e 'registered' para evitar dataleak
#df<- dados %>% dplyr::select(-c(casual,registered))
#skim(df)

# transformando em fatores variáveis categoricas que estao como numericas
df <- dados %>%
  mutate(season = as.factor(season),
         holiday = as.factor(holiday),
         weekday = as.factor(weekday),
         workingday = as.factor(workingday))
glimpse(df)

## analisando as variaveis categoricas e realizando ajustes se necessario

categorical_variables <- df %>%
  select_if(is.factor) %>%
  names()

for (i in categorical_variables) {
  cat("Variable:", i, "\n")
  print(table(dados[[i]]))
  cat("\n")
}

# Variavel Weathersit tem uma categoria com apenas 1 dado nela,
# iremos retirar pois não será possível realizar a regressão dessa forma

df %>% nrow()
df <- df %>% filter(weathersit != "heavy rain/snow")
df %>% nrow()


# Training - test split ---------------------------------------------------

set.seed(321)

split <- initial_split(df, prop = 0.8)
treinamento <- training(split)
teste <- testing(split)


# Tibble de resultados ----------------------------------------------------

fitted <- tibble(.pred = numeric(0),
                 observado = numeric(0),
                 modelo = character(0))

# Transformação de dados - formato usual -------------------------

# Transformação de dados - formato tidymodels --------------------

#faz a receita e prepara
(receita <- recipe(bikers ~ ., data = treinamento) %>%
   #step_log(bikers) %>% #transforma em ln a variavel objetivo
   step_rm(casual, registered) %>% #remove as 2 (sem sentido para o modelo)
   #step_meaninput() %>%
   #step_medianinput() %>%
   #step_modeinput() %>%
   step_zv(all_predictors()) %>% #remove variáveis que contém um único valor
   #step_discretize(season, holiday, weekday, workingday, options = list(cuts = 4)) %>% #transformo dbl em fct # nolint: line_length_linter.
   #step_num2factor(season, holiday, weekday, workingday)  %>%
   step_normalize(all_numeric(), -all_outcomes()) %>%  #normalizando as variaveis # nolint: line_length_linter.
   step_dummy(all_nominal(), -all_outcomes()))

(receita_prep <- prep(receita))

# bake na base de teste e treino
# obtem os dados de treinamento processados
treinamento_proc <- bake(receita_prep, new_data = NULL)
# obtem os dados de teste processados
teste_proc <- bake(receita_prep, new_data = teste)

# checando os dados processados
skim(treinamento_proc)
skim(teste_proc)

# Modelo1 - formato usual -------------------------------------------------





# Modelo 1 - formato tidymodels -------------------------------------------







# XGBoost - formato antigo-------------------------------------------------



# XGBoost - formato tidymodels --------------------------------------------

boost <- boost_tree(trees = tune(), learn_rate = tune(), mtry = tune(),
                    tree_depth = 2, min_n = tune(), sample_size = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# cv
set.seed(123)
cv_split <- vfold_cv(treinamento, v = 10)

# otimização de hiperparametro
doParallel::registerDoParallel(makeCluster(16)) #colocar nro de cores da maquina

tempo <- system.time({
  set.seed(123)
  boost_grid <- tune_grid(boost,
                          receita,
                          resamples = cv_split,
                          grid = 200,
                          metrics = metric_set(rmse))
})

autoplot(boost_grid) # plota os resultados do grid search

boost_grid %>%
  collect_metrics()  %>%   # visualiza o tibble de resultados
  arrange(mean)

(best_xgb <- boost_grid %>%
   select_best("rmse")) # salva o melhor conjunto de parametros

#faz o fit do modelo com o melhor conjunto de hiperparâmetros
boost_fit <- finalize_model(boost, parameters = best_xgb) %>%
  fit(bikers ~ ., teste_proc)

# tibble de predições
fitted_xgb <- boost_fit %>%
  predict(new_data = teste_proc) %>%
  mutate(observado = teste_proc$bikers,
         modelo = "xgboost tune - tidymodels")

# visualizando variáveis mais importantes
vip(boost_fit)

#plot predito vs realizado
fitted_xgb %>%
  ggplot(aes(x = observado, y = .pred)) +
  geom_point(size = 1, col = "blue") +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              linetype = "dashed",
              linewidth = 0.7) +
  labs(y = "predito xgboost", x = "Observado") +
  theme_bw()

# adicionando ao tibble de resultados
fitted <-  fitted %>% 
  bind_rows(fitted_xgb)

# Resultados e comentários finais -----------------------------------------

fitted %>% 
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) %>% 
  filter(.metric=='rmse')


tempo
#resultado de teste:
#tempo = 517s tree_deepth=tune()  grid=50  MSE=53
#tempo = 707s tree_deepth=2       grid=200 MSE=23.6