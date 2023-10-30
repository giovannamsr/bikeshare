
  
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
library(ranger)
library(randomForest)
library(xgboost)
library(pROC)

# input de dados ----------------------------------------------------------

dados <- Bikeshare

dados %>% skim() # não há dados faltantes, bikers possui distribuição exponencial # nolint
dados %>% glimpse() # passar pra fator: season, holiday, weekday, workingday
# e day?

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

#copia das bases de teste e treinamento
treinamento_usual <- treinamento
teste_usual <- teste

# Remover as colunas 'casual' e 'registered'
treinamento_usual <- treinamento_usual %>%
  select(-casual, -registered)
teste_usual <- teste_usual %>%
  select(-casual, -registered)

# Remover variáveis que contém apenas 1 valor
treinamento_usual <- treinamento_usual %>%
  select_if(~!all(is.na(.)) & length(unique(.)) > 1)
teste_usual <- teste[, colnames(treinamento_usual)]

# Normalizar as variáveis numéricas
numeric_columns <- names(which(sapply(treinamento_usual, is.numeric)))
treinamento_usual[numeric_columns] <- scale(treinamento_usual[numeric_columns])
teste_usual[numeric_columns] <- scale(teste_usual[numeric_columns])

# Criar variáveis dummy para variáveis nominais
nominal_columns <- names(Filter(function(x) is.factor(x) || is.character(x), treinamento_usual))
treinamento_usual <- treinamento_usual %>%
  mutate(across(all_of(nominal_columns), as.factor)) %>%
  model.matrix(~ . - 1, data = .) %>%
  as.data.frame()

teste_usual <- teste_usual %>%
  mutate(across(all_of(nominal_columns), as.factor)) %>%
  model.matrix(~ . - 1, data = .) %>%
  as.data.frame()

#ajuste de colunas
treinamento_usual <- subset(treinamento_usual, select = -season1)
teste_usual <- subset(teste_usual, select = -season1)
col_usual <- colnames(treinamento_usual)
new_col_names_usual <- gsub("[ /]", "_", col_usual)
colnames(treinamento_usual) <- new_col_names_usual
colnames(teste_usual) <- new_col_names_usual

# Checar os dados processados
summary(treinamento_usual)
summary(teste_usual)


# Transformação de dados - formato tidymodels --------------------

#faz a receita e prepara
(receita <- recipe(bikers ~ ., data = treinamento) %>%
   #step_log(bikers) %>% #Rodamos o modelo com e sem LN e os resultados finais ficaram melhor sem o LN
   step_rm(casual, registered) %>% #remove as 2 (evita data leak, o somatorio das 2 é igual a variavel objetivo)
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

# Floresta Aleatoria - formato antigo -------------------------------------------------
(rf <- ranger(bikers ~ ., 
              num.trees = 500,
              #mtry = 3,
              #min.node.size = 70,
              #max.depth = 15,
              data = treinamento_usual,
              classification = FALSE))

# Floresta Aleatoria - formato tidymodels -------------------------------------------

# Definir os hiperparâmetros para busca
rf <- rand_forest(trees = tune(), mtry = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

# CV
set.seed(123)
cv_split <- vfold_cv(treinamento_proc, v = 10)

# Otimização de hiperparâmetros
doParallel::registerDoParallel(makeCluster(8))  # Defina o número de núcleos apropriado

tempo <- system.time({
  set.seed(123)
  rf_grid <- tune_grid(rf,
                       recipe(bikers ~ ., data = treinamento_proc),
                       resamples = cv_split,
                       grid = 200,
                       metrics = metric_set(rmse))
})

autoplot(rf_grid)  # Plota os resultados do grid search

rf_grid %>%
  collect_metrics() %>%   # Visualiza o tibble de resultados
  arrange(mean)

# Seleciona o melhor conjunto de hiperparâmetros
best_rf <- rf_grid %>%
  select_best("rmse")

# Faz o fit do modelo com o melhor conjunto de hiperparâmetros
rf_fit <- finalize_model(rf, parameters = best_rf) %>%
  fit(bikers ~ ., data = teste_proc)

# Tibble de predições
fitted_rf <- rf_fit %>%
  predict(new_data = teste_proc) %>%
  mutate(observado = teste_proc$bikers,
         modelo = "Random Forest - tidymodels")

# XGBoost - formato antigo-------------------------------------------------


#criar matrix

dtrain <- xgb.DMatrix(label = treinamento_usual$bikers, 
                      data = as.matrix(select(treinamento_usual, -bikers))) #transformar a base de treino em matrix

dtest <- xgb.DMatrix(label = teste_usual$bikers,
                     data = as.matrix(select(teste_usual, -bikers)))  #transformar a base de teste em matrix

(fit_xgb <- xgb.train(data = dtrain, nrounds = 100, max_depth = 1, eta = 0.3,
                      nthread = 3, verbose = FALSE, objective = "reg:squarederror"))

importancia <- xgb.importance(model = fit_xgb)
xgb.plot.importance(importancia, rel_to_first = TRUE, top_n = 10, xlab = "Relative Import")


pred_xgb <- predict(fit_xgb, dtest)
sqrt(mean((pred_xgb - teste_usual$bikers)^2))


ajusta_bst <- function(splits, eta, nrounds, max_depth) {
  dtrain <- xgb.DMatrix(label = treinamento_usual$bikers,
                          data = as.matrix(select(treinamento_usual, -bikers)))
  dtest <- xgb.DMatrix(label = teste_usual$bikers,
                         data = as.matrix(select(teste_usual, bikers)))
  fit <- xgb.train(data = dtrain, nrounds = nrounds, max_depth = max_depth, eta = eta,
                   nthread = 3, verbose = FALSE, objective = "reg:squarederror")
  eqm <- mean((teste_usual$bikers - predict(fit, as.matrix(select(teste_usual, -bikers))))^2)
  return(sqrt(eqm))
}

set.seed(123)
hiperparametros <- crossing(eta = c(.01, .1),
                            nrounds = c(250, 750),
                            max_depth = c(1, 4))
resultados <- rsample::vfold_cv(treinamento_usual, 5) %>%
  crossing(hiperparametros) %>%
  mutate(reqm = pmap_dbl(list(splits, eta, nrounds, max_depth), ajusta_bst))
resultados %>%
  group_by(eta, nrounds, max_depth) %>%
  summarise(reqm = mean(reqm)) %>%
  arrange(reqm)

fit_xgb <- xgb.train(data = dtrain, nrounds = 750, max_depth = 2, eta = 0.3,
                     nthread = 3, verbose = FALSE, objective = "reg:squarederror")
pred_xgb <- predict(fit_xgb, dtest)
sqrt(mean((teste_usual$bikers - pred_xgb)^2))

fit_lm <- lm(bikers ~ ., treinamento_usual)
pred_lm <- predict(fit_lm, teste_usual)
sqrt(mean((teste_usual$bikers - pred_lm)^2))
  
# XGBoost - formato tidymodels --------------------------------------------

boost <- boost_tree(trees = tune(), learn_rate = tune(), mtry = tune(),
                    tree_depth = tune(), min_n = tune(), sample_size = tune()) %>%
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


