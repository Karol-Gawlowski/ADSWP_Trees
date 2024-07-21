source("init.R")
source("Models/XGBoost.R")
source("Models/train_GLM_w_XGB.R")
source("Models/train_multipl_GLM_XGB.R")
source("Models/train_GLM_XGBoosted.R")
source("Models/SAV_GLM.R")
# source("Models/stacked_ensemble.R")

CV = 5

set.seed(1)

CV_vec = sample(1:CV,replace = T,size = nrow(dt_list$fre_mtpl2_freq))

models = list()
results = list()
losses = data.frame(CV = paste0("CV_",1:CV),
                    homog = NA,
                    glm = NA,
                    SAV_glm = NA,
                    XGB = NA,
                    train_GLM_w_XGB = NA,
                    GLM_XGB = NA,
                    multipl_GLM_XGB = NA,
                    SAV_GLM_XGB_model = NA,
                    multipl_SAV_GLM_XGB = NA,
                    XGB_init_GLM =NA)

for (i in 1:CV){
  
  # train_rows = which(CV_vec != i & CV_vec != max(CV))
  # valid_rows = which(CV_vec == max(CV))
  
  valid_rows = which(CV_vec == i)
  test_i = (i %% 5) + 1
  test_rows = which(CV_vec == test_i)
  train_rows = which(CV_vec != i & CV_vec != test_i)
  
  iter = paste0("CV_",i)
  
  models[[iter]] = list()
  
  results[[iter]] = data.frame(ID = dt_list$fre_mtpl2_freq$IDpol[test_rows],
                               actual = dt_list$fre_mtpl2_freq$ClaimNb[test_rows],
                               glm = NA,
                               SAV_glm = NA,
                               XGB = NA,
                               train_GLM_w_XGB = NA,
                               GLM_XGB = NA,
                               multipl_GLM_XGB = NA,
                               SAV_GLM_XGB_model = NA,
                               multipl_SAV_GLM_XGB = NA,
                               XGB_init_GLM =NA) %>% 
    mutate(homog = mean(dt_list$fre_mtpl2_freq$ClaimNb[train_rows]))
  
  # homogenous model ------------------------------------------------- 
  
  info_helper(n=paste0(iter," homog"))
  
  losses$homog[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                     y_pred = results[[iter]]$homog)
  
  # glm ------------------------------------------------- 
  
  info_helper(n=paste0(iter," glm"))
  
  models[[iter]]$glm_model = glm(formula = ClaimNb~.,
                                 family = poisson,
                                 data = dt_list$fre_mtpl2_freq[valid_rows,-c(1,3)])
  
  results[[iter]]$glm = as.vector(predict(models[[iter]]$glm_model,
                                          dt_list$fre_mtpl2_freq[test_rows,-c(1,3)],type="response"))
  
  losses$glm[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                   y_pred = results[[iter]]$glm)
  
  # SAV glm ------------------------------------------------- 
  
  info_helper(n=paste0(iter," SAV glm"))
  
  models[[iter]]$SAV_glm_model = SAV_glm(data = dt_list$fre_mtpl2_freq[train_rows,-c(1,3)])
  
  results[[iter]]$SAV_glm = predict(models[[iter]]$SAV_glm_model,dt_list$fre_mtpl2_freq[test_rows,-c(1,3)])
  
  losses$SAV_glm[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                       y_pred = results[[iter]]$SAV_glm)
  
  # XGBoost  -------------------------------------------
  
  info_helper(n=paste0(iter," base XGB"))
  
  models[[iter]]$XGB_model = train_XGBoost(dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                           y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                           vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                      y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows]))
  
  results[[iter]]$XGB = predict(models[[iter]]$XGB_model,
                                xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])))
  
  losses$XGB[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                   y_pred = results[[iter]]$XGB)
  
  # GLM with XGB pred as covariate  -------------------------------------------
  
  info_helper(n=paste0(iter," glm with XGB covariate"))
  
  models[[iter]]$train_GLM_w_XGB = train_GLM_w_XGB(xgb_model = models[[iter]]$XGB_model,
                                                   dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                   y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows])
  
  results[[iter]]$train_GLM_w_XGB = predict(models[[iter]]$train_GLM_w_XGB,
                                            dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])
  
  losses$train_GLM_w_XGB[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                               y_pred = results[[iter]]$train_GLM_w_XGB)
  
  
  # GLM + XGBoost  -------------------------------------------
  
  info_helper(n=paste0(iter," Boosted GLM"))
  
  models[[iter]]$GLM_XGB_model = train_GLM_XGBoosted(glm_model = models[[iter]]$glm_model,
                                                     dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                     y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                     vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                                y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows]))
  
  results[[iter]]$GLM_XGB = predict(models[[iter]]$GLM_XGB_model,
                                    dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])
  
  
  losses$GLM_XGB[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                       y_pred = results[[iter]]$GLM_XGB)
  
  # GLM * XGBoost  -------------------------------------------
  
  info_helper(n=paste0(iter," glm with XGB multipl"))
  
  models[[iter]]$multipl_GLM_XGB = train_multipl_GLM_XGB(glm_model = models[[iter]]$glm_model,
                                                         dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                         y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                         vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                                    y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows])
  )
  
  results[[iter]]$multipl_GLM_XGB = predict(models[[iter]]$multipl_GLM_XGB,
                                            dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])
  
  losses$multipl_GLM_XGB[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                               y_pred = results[[iter]]$multipl_GLM_XGB)
  
  # SAV_GLM + XGBoost  -------------------------------------------
  
  info_helper(n=paste0(iter," Boosted SAV GLM"))
  
  models[[iter]]$SAV_GLM_XGB_model = train_GLM_XGBoosted(glm_model = models[[iter]]$SAV_glm_model,
                                                         dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                         y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                         vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                                    y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows]))
  
  results[[iter]]$SAV_GLM_XGB_model = predict(models[[iter]]$SAV_GLM_XGB_model,
                                              dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])
  
  losses$SAV_GLM_XGB_model[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                                 y_pred = results[[iter]]$SAV_GLM_XGB_model)
  
  # SAV_GLM * XGBoost  -------------------------------------------
  
  info_helper(n=paste0(iter,"SAV glm with XGB multipl"))
  
  models[[iter]]$multipl_SAV_GLM_XGB = train_multipl_GLM_XGB(glm_model = models[[iter]]$SAV_glm_model,
                                                             dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                             y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                             vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                                        y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows])
  )
  
  results[[iter]]$multipl_SAV_GLM_XGB = predict(models[[iter]]$multipl_SAV_GLM_XGB,
                                                dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])
  
  losses$multipl_SAV_GLM_XGB[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                                   y_pred = results[[iter]]$multipl_SAV_GLM_XGB)
  
  # Stacked ensemble  -------------------------------------------

  # models[[iter]]$stacked_ensemble = train_ff_nn(dt = matrix(data = c(predict(models[[iter]]$SAV_glm_model,
  #                                                                            dt_list$fre_mtpl2_freq[train_rows,-c(1,3)]),
  #                                                                    predict(models[[iter]]$XGB_model,
  #                                                                            xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)])))),
  #                                                           ncol = 2,
  #                                                           byrow = F),
  #                                               y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
  #                                               vdt = list(x_val = matrix(data = c(predict(models[[iter]]$SAV_glm_model,
  #                                                                                          dt_list$fre_mtpl2_freq[valid_rows,-c(1,3)]),
  #                                                                                  predict(models[[iter]]$XGB_model,
  #                                                                                          xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)])))),
  #                                                                         ncol = 2,
  #                                                                         byrow = F),
  #                                                          y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows]),
  #                                               n_dropout = c(F,F),
  #                                               n_layers = 2,
  #                                               n_units = c(20,15),
  #                                               n_activ = c("sigmoid","sigmoid"),
  #                                               # n_activ = c("tanh","tanh","tanh"),
  #                                               lr = 0.0025,
  #                                               bs = 2^13,
  #                                               ep = 100)
  # 
  # results[[iter]]$stacked_ensemble = predict(models[[iter]]$stacked_ensemble,
  #                                            matrix(data = c(predict(models[[iter]]$SAV_glm_model,
  #                                                                    dt_list$fre_mtpl2_freq[test_rows,-c(1,3)]),
  #                                                            predict(models[[iter]]$XGB_model,
  #                                                                    xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)])))),
  #                                                   ncol = 2,
  #                                                   byrow = F))[,1]
  # 
  # losses$stacked_ensemble[i] = poisson_deviance(y_true = results[[iter]]$actual,
  #                                               y_pred = results[[iter]]$stacked_ensemble)
  # 
  
  # XGBoost intialised with GLM  -------------------------------------------
  
  # info_helper(n=paste0(iter," XGB init GLM"))
  # 
  # models[[iter]]$XGB_init_GLM_model = train_XGBoost(dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
  #                                                   y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
  #                                                   vdt = list(x_val = dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)],
  #                                                              y_val = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows]),
  #                                                   use_glm= TRUE
  # )
  # 
  # results[[iter]]$XGB_init_GLM = predict(models[[iter]]$XGB_init_GLM_model,
  #                                        xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)])))

  # XGBoost intialised with GLM  -------------------------------------------
  
  info_helper(n=paste0(iter," XGB init GLM"))
  
  models[[iter]]$XGB_init_GLM_model = train_XGBoost(glm_model = models[[iter]]$glm_model, dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                    y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                    vdt = list(x_val = dt_list$fre_mtpl2_freq[valid_rows,-c(1,2,3)],
                                                               y_val = dt_list$fre_mtpl2_freq$ClaimNb[valid_rows])
  )
  
  dval_with_margin = xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)]))
  val_base_margin = predict(models[[iter]]$glm_model, dt_list$fre_mtpl2_freq[test_rows,-c(1,2,3)],type="link")
  setinfo(dval_with_margin, "base_margin", val_base_margin)
  
  results[[iter]]$XGB_init_GLM = predict(models[[iter]]$XGB_init_GLM_model,dval_with_margin)
  
  losses$XGB_init_GLM[i] = poisson_deviance(y_true = results[[iter]]$actual,
                                            y_pred = results[[iter]]$XGB_init_GLM)  

}

sink(NULL)

saveRDS(list(losses = losses,
             results = results),file = "Results/The_Actuary_trees_wo_models_v12_VALIDATION.rds")

analysis = bind_rows(results,.id = "id")  %>% 
  select(id,actual,homog,glm,SAV_glm,XGB,train_GLM_w_XGB,GLM_XGB,
         multipl_GLM_XGB,SAV_GLM_XGB_model,multipl_SAV_GLM_XGB) %>% 
  pivot_longer(cols = homog:multipl_SAV_GLM_XGB) %>% 
  mutate(actual = actual,
         value = value,
         poiss = Vectorize(poisson_deviance)(y_true = actual,
                                             y_pred = value)) 

poiss_per_CV = rbind(losses,
                     losses %>%
                       pivot_longer(cols = !CV) %>%
                       group_by(name) %>%
                       summarise(mean_poiss = mean(value)) %>%
                       arrange(mean_poiss) %>%
                       pivot_wider(values_from = mean_poiss,names_from = name) %>%
                       mutate(CV = "mean_poiss"))

# pinball
poiss_per_CV %>% 
  mutate_if(is.numeric,~ if_else(. == homog, .,1 -  ./homog)) %>% 
  select(-homog) %>% 
  mutate_if(is.numeric,scales::percent,0.1)



poiss_dens(an = analysis,m = c("glm","multipl_GLM_XGB","GLM_XGB"))

poiss_dens(an = analysis,m = c("glm","multipl_GLM_XGB","multipl_SAV_GLM_XGB"))
poiss_dens(an = analysis,m = c("glm","GLM_XGB","SAV_GLM_XGB_model"))
