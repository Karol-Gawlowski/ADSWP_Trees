train_GLM_XGBoosted = function(glm_model,
                               dt,
                               y,
                               vdt){
  
  glm_preds = predict(glm_model,(dt))
  
  Residuals_train = residuals(glm_model, type="response")
  
  Residuals_val = vdt$y_val-
    as.vector(predict(glm_model, vdt$x_val,type="response"))
  
  
  GLM_XGB_model = train_XGBoost(dt,
                                y = Residuals_train,
                                vdt = list(x_val = dt_list$fre_mtpl2_freq[-train_rows,-c(1,2,3)],
                                           y_val = Residuals_val),
                                objective = "reg:squarederror",
                                eval_metric = "rmse",
                                eta = 0.005,
                                max_depth = 5,
                                tweedie_variance_power = 0
                                
  )
  
  
  toreturn = list(glm_model = glm_model, GLM_XGB_model = GLM_XGB_model)
  
  class(toreturn) = "train_GLM_XGBoosted" 
  
  return(toreturn)
  
}

predict.train_GLM_XGBoosted = function(model,dt){
  
  
  xgb_preds = predict(Model$Residual_Model$GLM_XGB_model,xgb.DMatrix(data.matrix(dt)), type="response")
  glm_preds = predict(Model$Base_Modeldt)
  
  results = pmax(0,xgb_preds + glm_preds)
  
  return(xgb_preds)
  
}
