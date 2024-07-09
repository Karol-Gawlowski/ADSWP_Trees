train_GLM_XGBoosted = function(glm_model,
                               dt,
                               y,
                               vdt){
  
  Residuals_train = y - as.vector(predict(glm_model, dt,type="response")) 
  
  Residuals_val = vdt$y_val - as.vector(predict(glm_model, vdt$x_val,type="response"))
  
  GLM_XGB_model = train_XGBoost(dt,
                                y = Residuals_train,
                                vdt = list(x_val = vdt$x_val,
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
  
  xgb_preds = predict(model$GLM_XGB_model,xgb.DMatrix(data.matrix(dt)), type="response")
  glm_preds = predict(model$glm_model,dt, type="response")
  
  results = pmax(0,xgb_preds + glm_preds)
  
  return(results)
  
}
