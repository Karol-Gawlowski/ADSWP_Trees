train_GLM_w_XGB = function(xgb_model,
                           dt,
                           y){
  
  xgb_preds = predict(xgb_model,xgb.DMatrix(data.matrix(dt)))
  
  glm_model = glm(formula = y~.,
                  family = poisson,
                  data = cbind(y,dt,xgb_preds))
 
  toreturn = list(glm_model = glm_model,xgb_model = xgb_model)
  
  class(toreturn) = "train_GLM_w_XGB" 
  
  return(toreturn)
  
}

predict.train_GLM_w_XGB = function(model,dt){
  
  xgb_preds = predict(model$xgb_model,xgb.DMatrix(data.matrix(dt)))
  
  return(predict(model$glm_model,cbind(dt,xgb_preds),type="response"))
  
}
