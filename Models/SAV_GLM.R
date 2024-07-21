# SAV study on GLM

SAV_preproc = function(dt){
  
  dt %>% 
    mutate(
      # ClaimNb = pmin(as.integer(ClaimNb), 4),
           VehAge = pmin(VehAge, 20),
           DrivAge = pmin(DrivAge, 90),
           BonusMalus = pmin(BonusMalus, 150),
           Density = round(log(Density), 2),
           VehGas = factor(VehGas),
           # Exposure = pmin(Exposure, 1)
           
           AreaGLM = as.factor(Area),
           VehPowerGLM = as.factor(pmin(VehPower, 9)),
           VehAgeGLM = cut(VehAge, breaks = c(-Inf, 0, 10, Inf), labels = c("1","2","3")),
           DrivAgeGLM = cut(DrivAge, breaks = c(-Inf, 20, 25, 30, 40, 50, 70, Inf), labels = c("1","2","3","4","5","6","7")),
           BonusMalusGLM = as.integer(pmin(BonusMalus, 150)),
           DensityGLM = as.numeric(Density),
           VehAgeGLM = relevel(VehAgeGLM, ref = "2"),
           DrivAgeGLM = relevel(DrivAgeGLM, ref = "5"),
           Region = relevel(factor(Region), ref = "R24")
    ) %>% 
    return()
  
}

SAV_glm = function(data){
  
  glm1 <- glm(
    ClaimNb ~ VehPowerGLM + VehAgeGLM + DrivAgeGLM + BonusMalusGLM + VehBrand +
      VehGas + DensityGLM + Region + AreaGLM,
    
    data =   data %>% SAV_preproc,
    # offset = log(Exposure), 
    family = poisson()
  )
  
  class(glm1) = "SAV_glm"
  
  return(glm1)
  
}


predict.SAV_glm = function(model,data,type="response"){
  
  # browser()
  
  class(model) = "glm"
  
  as.vector(predict.glm(model,data %>% SAV_preproc,type=type))
  
}



