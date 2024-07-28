library(tidyverse)
library(xgboost)
library(reticulate)
library(tensorflow)
library(keras)
library(mlrMBO)

slice = dplyr::slice
rename = dplyr::rename

seed = 2
set.seed(seed)

# load data objects
dt_list = list()

# Data
dt_list$fre_mtpl2_freq = read.csv("freMTPL2freq.csv") %>%  

  mutate(  Exposure = pmin(1, Exposure),
           ClaimNb = pmin(15, ClaimNb / Exposure)
           ) %>% 
  slice(sample(1:nrow(.),replace = F))

#Poisson Deviance - Loss function
poiss_loss = function(y_true,y_pred){
  y_true*log(y_true/y_pred)-(y_true-y_pred)
}

info_helper = function(n,t="misc/sink.txt"){
  
  sink(NULL)
  cat(paste0(n,"\n"))
  sink(t)
  
}


poiss_dens = function(an = analysis,
                      m = c("multipl_SAV_GLM_XGB","multipl_GLM_XGB")){

  an %>%
    # filter(name!="homog") %>%
    filter(name %in% m) %>%
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
  
}

# Poisson Deviance 
custom_poisson <- function( y_true, y_pred ) {
  # Mario V. Wüthrich , Michael Merz
  # Statistical Foundations of Actuarial Learning and its Applications
  # Table 4.1 page 87
  
  # 2 * (y_pred - y_true - y_true*log(y_pred/y_true))
  # 2 (μ − y − ylog(μ/y))
  
  K <- backend()
  
  K$mean(2 * (y_pred - y_true - y_true * K$log((y_pred+10^-7)/(y_true+10^-7))))
  
}


poisson_deviance = function(y_true,y_pred,keras=F,correction = +10^-7){
  
  # stopifnot(length(true)!=length(pred),"different input lengths!")
  
  if(keras){
    # keras:
    # pd = y_pred - y_true * log(y_pred)
    
  }else{
    
    pd =  mean((y_pred - y_true - y_true * log((y_pred+correction)/(y_true+correction))))
    
  }
  
  return(2 * pd)
  
}


multiple_lift = function(y_true,
                          y_pred_df,
                          tiles = 10){
  
  tiles_list = list()
  
  for (i in colnames(y_pred_df)){
    
    tiles_list[[i]] = data.frame(model = y_pred_df[[i]],
                                 actual = y_true) %>% 
      mutate(tiles = ntile(model,tiles)) %>%
      group_by(tiles) %>% 
      summarise(model = mean(model)) %>% 
      pull(model)
  }
  
  bind_cols(tiles_list) %>% 
    mutate(t = 1:tiles) %>% 
    set_names(c(colnames(y_pred_df),"tiles")) %>% 
    pivot_longer(cols = !tiles) %>% 
    ggplot(aes(x = tiles,y=value,group=name,color=name,linetype=name))+
    geom_point()+
    geom_line()
  
}


double_lift = function(an,
                       actual,
                       m1,
                       m2,
                       tiles=5){
  
  data.frame(actual = an[[actual]],
             model1 = an[[m1]],
             model2 = an[[m2]]
             ) %>% 
    group_by(tiles = factor(ntile(model1/model2,tiles),levels=1:10)) %>% 
    summarise(model1 = mean(model1),
              act = mean(actual),
              model2 = mean(model2)
              ) %>% 
    pivot_longer(cols = !tiles) %>% 
    mutate(name = case_when(name=="model1" ~ m1,
                            name=="model2" ~ m2,
                            TRUE ~ "actual"
                            ),
           name = factor(name,levels = c(m1,"actual",m2))
           ) %>% 
    ggplot(aes(x = tiles,y=value,fill=name))+
    geom_col(position = "dodge")+
    scale_fill_manual(values = c("red","yellow","blue"))
  
}





one_way_chart = function(dt = analysis_wide, #wide format
                         models = c("glm","ff_nn"), # string vector
                         y = "actual", # actual string
                         xvar = "DrivAge", # string
                         expo = "Exposure",# string
                         buckets = 10 # int for numerical
){
  
  if(is.numeric(dt[[xvar]])){
    
    tot_ex = sum(dt[[expo]])
    
    dt = dt %>% 
      arrange(!!sym(xvar)) %>% 
      mutate(!!sym(xvar) := cut(cumsum(!!sym(expo)),breaks = seq(0,tot_ex,length.out = buckets)))
    
  }else if(!is.character(dt[[xvar]])){
    stop("x format wrong")
  }
  
  m = dt[c(xvar,y,models)] %>% 
    group_by(!!sym(xvar)) %>% 
    summarise_all(.funs = mean)
  
  e = dt[c(xvar,expo)] %>% 
    group_by(!!sym(xvar)) %>% 
    summarise(ex = sum(!!sym(expo)))
  
  prop = max(m[models])/max(e$ex)
  
  # IF IS STRING THEN ORDER BY EXPO SIZE
  
  # add secondary axis
  # add tilt if numeric
  
  ggplot()+
    geom_point(data = m %>% 
                 pivot_longer(cols = c(models,y),names_to = "model"),
               aes_string(x = xvar,y = "value",color = "model",shape = "model"),
               alpha = 0.8,
               size = 2)+
    geom_col(data = e %>% 
               mutate(ex = ex*prop*0.5),mapping = aes_string(x = xvar,y="ex"),
             alpha = 0.3) +
    # ggdark::dark_theme_classic()+
    # theme(panel.grid.minor = element_line(colour="darkgrey", size=0.01,linetype = 3))+
    theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))+ 
    scale_y_continuous(
      sec.axis = sec_axis(~ . / prop, name = "Exposure",labels = scales::comma)
    )+
    ggtitle(paste0("One way - ",xvar))+
    ylab("Freq")+
    xlab(xvar) %>% 
    return()
  
}
