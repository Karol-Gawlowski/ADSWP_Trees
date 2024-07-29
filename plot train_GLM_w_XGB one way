
# plot ---------------------------------------

# one way chart for xgb predictor

library(ggplot2)
library(dplyr)
library(patchwork)

# Function to process each fold
process_fold <- function(df, fold) {
  df <- df %>%
    arrange(train_GLM_w_XGB) %>%
    mutate(bucket = cut_number(train_GLM_w_XGB, n = 20)) %>%
    group_by(bucket) %>%
    summarise(mean_train_GLM_w_XGB = mean(train_GLM_w_XGB),
              mean_actual = mean(actual),
              mean_glm = mean(glm),
              count = n())
  
  df$fold <- fold
  return(df)
}

# Process all folds and combine the results
processed_results <- bind_rows(
  lapply(1:length(results), function(i) process_fold(results[[i]], paste0("CV_", i)))
)

# Function to create line plot for each fold
create_line_plot <- function(data) {
  ggplot(data, aes(x = mean_train_GLM_w_XGB)) +
    geom_line(aes(y = mean_actual, color = "Actual")) +
    geom_point(aes(y = mean_actual, color = "Actual")) +
    geom_line(aes(y = mean_glm, color = "GLM")) +
    geom_point(aes(y = mean_glm, color = "GLM")) +
    labs(x = "Train_GLM_w_XGB (Binned)",
         y = "Mean Value") +
    theme_minimal() +
    scale_color_manual(name = "Legend", values = c("Actual" = "blue", "GLM" = "red"))
}

# Function to create bar plot for each fold
create_bar_plot <- function(data) {
  ggplot(data, aes(x = mean_train_GLM_w_XGB, y = count)) +
    geom_bar(stat = "identity", fill = "grey", alpha = 0.7) +
    labs(x = "Train_GLM_w_XGB (Binned)",
         y = "Count") +
    theme_minimal()
}

# Create line plots for each fold
line_plots <- lapply(unique(processed_results$fold), function(fold) {
  fold_data <- processed_results %>% filter(fold == !!fold)
  create_line_plot(fold_data) + ggtitle(paste("Line Plot -", fold))
})

# Create bar plots for each fold
bar_plots <- lapply(unique(processed_results$fold), function(fold) {
  fold_data <- processed_results %>% filter(fold == !!fold)
  create_bar_plot(fold_data) + ggtitle(paste("Bar Plot -", fold))
})

# Combine line plots into one window (3 plots on top row, 2 on bottom row)
combined_line_plots <- wrap_plots(line_plots, ncol = 3, nrow = 2)

# Combine bar plots into one window (3 plots on top row, 2 on bottom row)
combined_bar_plots <- wrap_plots(bar_plots, ncol = 3, nrow = 2)

# Print the combined plots
print(combined_line_plots)
print(combined_bar_plots)

