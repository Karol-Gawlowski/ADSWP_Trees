train_XGBoost = function(dt,
                         y,
                         vdt,
                         eta = 0.3, # low eta value means model more robust to overfitting but slower to compute
                         gamma = 0, # minimum loss reduction required to make a further partition on a leaf node of the tree.
                         max_depth = 6 , # maximum depth of a tree
                         min_child_weight = 1, # minimum sum of instance weight (hessian) needed in a child
                         subsample = 1, # subsample ratio of the training instance
                         colsample_bytree = 1, # colsample_bytree
                         objective = "count:poisson",
                         eval_metric = "poisson-nloglik",
                         tweedie_variance_power = 1.5,
                         use_glm = F,
                         glm_model = NULL,
                         default_params = T
){
  
  # #parameters list
  # params <- list(
  #   
  #   #function inputs
  #   eta=eta,
  #   gamma=gamma,
  #   max_depth=max_depth,
  #   min_child_weight=min_child_weight,
  #   subsample=subsample,
  #   colsample_bytree=colsample_bytree,
  #   tweedie_variance_power=tweedie_variance_power,
  #   
  #   #Fixed
  #   objective = objective,
  #   eval_metric = eval_metric
  # )
  
  if(default_params){
    
    params <- list(
      #Fixed
      objective = objective,
      eval_metric = eval_metric)
    
  }else{
    
    #parameters list
    params <- list(
      
      #function inputs
      eta=eta,
      gamma=gamma,
      max_depth=max_depth,
      min_child_weight=min_child_weight,
      subsample=subsample,
      colsample_bytree=colsample_bytree,
      tweedie_variance_power=tweedie_variance_power,
      
      #Fixed
      objective = objective,
      eval_metric = eval_metric)
    
  }
  
  #training set
  y_train <- y
  X_train <- data.matrix(dt)
  
  #train
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  vtrain <- xgb.DMatrix(data.matrix(vdt$x_val), label = vdt$y_val)
  
  # Initialize with GLM predictions if use_glm is TRUE
  if (use_glm && !is.null(glm_model)) {
    
    glm_predictions_train <- predict(glm_model, dt,type="link")
    setinfo(dtrain, "base_margin", unname(glm_predictions_train))
    glm_predictions_val <- predict(glm_model, vdt$x_val,type="link")
    setinfo(vtrain, "base_margin", unname(glm_predictions_val))
  }
  
  
  
  
  # Fit final, tuned model
  fit <- xgb.train(
    params = params, 
    data = dtrain, 
    nrounds = 1000,
    verbose = 1,
    watchlist = list(validation = vtrain),
    early_stopping_rounds = 10
    
  )
  
  if(use_glm && !is.null(glm_model)) {
    
    fit = list(glm_model = glm_model,xgb = fit)
    
    class(fit) = "base_margin_xgb"
  }
  
  return(fit) 
  
}

predict.base_margin_xgb = function(model,dt){

  dval_with_margin = xgb.DMatrix(data.matrix(dt))
  val_base_margin = predict(model$glm_model, dt,type="link")
  setinfo(dval_with_margin, "base_margin", unname(val_base_margin))
  
  return(predict(model$xgb,dval_with_margin))
  
}

