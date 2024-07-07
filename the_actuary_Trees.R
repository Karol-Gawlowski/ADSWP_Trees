source("init.R")
source("Models/XGBoost.R")
source("Models/train_GLM_w_XGB.R")

CV = 5

set.seed(1)

CV_vec = sample(1:CV,replace = T,size = nrow(dt_list$fre_mtpl2_freq))

models = list()
results = list()
losses = data.frame(CV = paste0("CV_",1:CV),
                    homog = NA,
                    glm = NA,
                    XGB = NA,
                    train_GLM_w_XGB = NA,
                    GLM_XGB = NA,
                    multipl_GLM_XGB = NA)

# fitted = readRDS("The_Actuary_final_results_wo_models_v5.rds")
# 
# losses = fitted$losses
# results = fitted$results

for (i in 1:CV){
  
  train_rows = which(CV_vec != i)
  
  models[[paste0("CV_",i)]] = list()
  
  results[[paste0("CV_",i)]] = data.frame(ID = dt_list$fre_mtpl2_freq$IDpol[-train_rows],
                                          actual = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows],
                                          glm = NA,
                                          XGB = NA,
                                          train_GLM_w_XGB = NA,
                                          GLM_XGB = NA,
                                          multipl_GLM_XGB = NA) %>% 
    mutate(homog = mean(dt_list$fre_mtpl2_freq$ClaimNb[train_rows]))
  
  encoder = preproc(dt_frame = dt_list$fre_mtpl2_freq[train_rows,],
                    y = "ClaimNb",
                    num = "norm",
                    cat = "ohe",
                    bypass = NULL,
                    exclude = c("IDpol","Exposure"),
                    verbose = T)
  
  train = encoder(dt_list$fre_mtpl2_freq[train_rows,])
  test = encoder(dt_list$fre_mtpl2_freq[-train_rows,])
  
  # homogenous model ------------------------------------------------- 
  
  losses$homog[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                     y_pred = results[[paste0("CV_",i)]]$homog)
  
  # glm ------------------------------------------------- 
  
  models[[paste0("CV_",i)]]$glm_model = glm(formula = ClaimNb~.,
                                            family = poisson,
                                            data = dt_list$fre_mtpl2_freq[train_rows,-c(1,3)])
  
  results[[paste0("CV_",i)]]$glm = as.vector(predict(models[[paste0("CV_",i)]]$glm_model,
                                                     dt_list$fre_mtpl2_freq[-train_rows,-c(1,3)],type="response"))
  
  losses$glm[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                   y_pred = results[[paste0("CV_",i)]]$glm)
  
  # XGBoost  -------------------------------------------
  
  models[[paste0("CV_",i)]]$XGB_model = train_XGBoost(dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                      y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                      vdt = list(x_val = dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)],
                                                                 y_val = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows])
  )
  
  results[[paste0("CV_",i)]]$XGB = predict(models[[paste0("CV_",i)]]$XGB_model,
                                           xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)])))
  
  losses$XGB[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                   y_pred = results[[paste0("CV_",i)]]$XGB)
  
  
  # GLM with XGB pred as covariate  -------------------------------------------
  
  models[[paste0("CV_",i)]]$train_GLM_w_XGB = train_GLM_w_XGB(xgb_model = models[[paste0("CV_",i)]]$XGB_model,
                                                              dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                              y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows])
  
  results[[paste0("CV_",i)]]$train_GLM_w_XGB = predict(models[[paste0("CV_",i)]]$train_GLM_w_XGB,
                                                       dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)])
  
  losses$train_GLM_w_XGB[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                               y_pred = results[[paste0("CV_",i)]]$train_GLM_w_XGB)
  
  
  # GLM + XGBoost  -------------------------------------------
  
  #vdt predictions
  Residuals_glm = residuals(models[[paste0("CV_",i)]]$glm_model, type="response")
  
  Residuals_val = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows]-
    as.vector(predict(models[[paste0("CV_",i)]]$glm_model, dt_list$fre_mtpl2_freq[-train_rows,-c(1,3)],type="response"))
  
  models[[paste0("CV_",i)]]$GLM_XGB_model = train_XGBoost(dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                          y = Residuals_glm,
                                                          vdt = list(x_val = dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)],
                                                                     y_val = Residuals_val),
                                                          objective = "reg:squarederror",
                                                          eval_metric = "rmse",
                                                          eta = 0.005,
                                                          max_depth = 5,
                                                          tweedie_variance_power = 0
                                                          
  )
  
  results[[paste0("CV_",i)]]$GLM_XGB = pmax(0,predict(models[[paste0("CV_",i)]]$GLM_XGB_model,
                                                      xgb.DMatrix(data.matrix(dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)]))) +
                                              results[[paste0("CV_",i)]]$glm)
  
  losses$GLM_XGB[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                       y_pred = results[[paste0("CV_",i)]]$GLM_XGB)
  
  
  # GLM * XGBoost  -------------------------------------------
  
  models[[paste0("CV_",i)]]$multipl_GLM_XGB = train_multipl_GLM_XGB(glm_model = models[[paste0("CV_",i)]]$glm_model,
                                                                    dt = dt_list$fre_mtpl2_freq[train_rows,-c(1,2,3)],
                                                                    y = dt_list$fre_mtpl2_freq$ClaimNb[train_rows],
                                                                    vdt = list(x_val = dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)],
                                                                               y_val = dt_list$fre_mtpl2_freq$ClaimNb[-train_rows])
  )
  
  results[[paste0("CV_",i)]]$multipl_GLM_XGB = predict(models[[paste0("CV_",i)]]$multipl_GLM_XGB,
                                                       dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)])
  
  losses$multipl_GLM_XGB[i] = poisson_deviance(y_true = results[[paste0("CV_",i)]]$actual,
                                               y_pred = results[[paste0("CV_",i)]]$multipl_GLM_XGB)
  
}


# save files
# saveRDS(list(losses = losses,
#              results = results,
#              models = models),file = "The_Actuary_trees_v1.rds")

saveRDS(list(losses = losses,
             results = results),file = "Results/The_Actuary_trees_wo_models_v5.rds")

# check calibration
bind_rows(results,.id = "id") %>% 
  select(-ID) %>% 
  group_by(id) %>% 
  summarise_all(mean)

# avg deviation from actual %
bind_rows(results,.id = "id") %>% 
  select(-ID) %>% 
  group_by(id) %>% 
  summarise_all(mean) %>% 
  mutate_if(is.numeric,~if_else(.==actual,.,./actual - 1)) %>% 
  select(-actual) %>% 
  mutate_if(is.numeric,scales::percent,0.1)

analysis = bind_rows(results,.id = "id")  %>% 
  select(id,actual,glm,XGB, homog, train_GLM_w_XGB, GLM_XGB,multipl_GLM_XGB) %>% 
  pivot_longer(cols = glm:multipl_GLM_XGB) %>% 
  mutate(actual = actual,
         value = value,
         poiss = Vectorize(poisson_deviance)(y_true = actual,
                                             y_pred = value)) 

# ovarall and per fold results
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
  

losses %>% 
  mutate_if(is.numeric,.funs = function(x)(x*c(data.frame(k=CV_vec) %>% 
                                                 count(k) %>% pull(n)))) %>% 
  janitor::adorn_totals()

analysis %>%
  filter(name!="homog") %>%
  # filter(name %in% c("train_GLM_w_XGB","multipl_GLM_XGB")) %>% 
  rename(model=name) %>% 
  ggplot(aes(x = poiss,fill=model,color=model,linetype=model))+
  geom_density(alpha=0.3,size=1)+
  ggplot2::scale_fill_manual(values = c("blue","yellow","green","red","white"))+
  xlim(0,0.75)+
  # facet_wrap(~name)+
  ggdark::dark_theme_classic()+
  # theme(panel.grid.minor = element_line(colour="darkgrey", size=0.01,linetype = 3))+
  ggtitle("Poisson deviance per observation, per model")+
  xlab("Poisson deviance")

# lift chart
multiple_lift(y_true = bind_rows(results,.id = "id") %>% pull(actual),
              y_pred_df = bind_rows(results,.id = "id") %>% select(glm,
                                                                   XGB,
                                                                   homog,
                                                                   GLM_XGB))+
  ggtitle("Combined lift chart")+
  xlab("Tiles")+
  ylab("Implied frequency")+
  ggdark::dark_theme_classic()

# remark on glm
merge(coefficients(models$CV_1$train_GLM_w_XGB$glm_model) %>% data.frame() %>% set_names("glm(xgb)") %>% rownames_to_column() ,
      coefficients(models$CV_1$glm_model) %>% data.frame()  %>% set_names("glm") %>% rownames_to_column(),
      by = "rowname",all.x=T) %>% mutate(diff_perc = scales::percent(`glm(xgb)`/glm - 1,0.1)) %>% mutate_at(vars(contains("glm")),scales::number,0.01)

