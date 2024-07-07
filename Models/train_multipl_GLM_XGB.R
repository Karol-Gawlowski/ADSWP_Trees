train_multipl_GLM_XGB = function(glm_model,
                                 dt,
                                 y,
                                 vdt){
  
  glm_preds_in = unname(predict(glm_model,dt,type="response"))
  glm_preds_out = unname(predict(glm_model,vdt$x_val,type="response"))
  
  xgb_model = train_XGBoost(dt = dt,
                            y = y/glm_preds_in,
                            vdt = list(x_val = vdt$x_val,
                                       y_val = vdt$y_val/glm_preds_out),
                            
                            objective = "count:poisson",
                            eval_metric = "poisson-nloglik" #,
                            # eta = 0.005,
                            # max_depth = 5,
                            # tweedie_variance_power = NULL
  )
  
  toreturn = list(glm_model = glm_model,xgb_model = xgb_model)
  
  class(toreturn) = "train_multipl_GLM_XGB" 
  
  return(toreturn)
  
}

predict.train_multipl_GLM_XGB = function(model,dt){

  glm = unname(predict(model$glm_model,dt,type="response"))
  xgb = predict(model$xgb_model,xgb.DMatrix(data.matrix(dt)),type="response")

  toreturn = glm * xgb
  
  if(any(is.na(toreturn) | is.nan(toreturn) | toreturn<0)){browser()}
    
  return(toreturn)
  
}

