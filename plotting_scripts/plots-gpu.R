
library(ggplot2)
library(dplyr)
require(gridExtra)

game <- 'MsPacman-v0'
beta<- '1.0_slow_decay'
path <- paste('/Users/alextsaptsinos/Documents/_Stanford/Courses/_2017Autumn/CS221/project/cs221project/data/aim_dql/',game,'/',beta,sep='')
setwd(path)
  
all_data <- read.csv("IM_worker_0.csv")
all_data$thread_id <- 0
for (i in 1:11){
  name <- paste(paste("IM_worker", i, sep = "_"), "csv", sep = ".")
  thread_data <- read.csv(name)
  thread_data$thread_id <- i
  all_data <- rbind(all_data, thread_data)   
}
all_data <- all_data %>% 
  mutate(thread_id = as.factor(thread_id))
all_data$global_frame <- all_data$global_frame / 1000000

p1 <- ggplot(all_data, aes(x = global_frame, 
                           y = extrinsic.reward)) +
  geom_smooth(se = TRUE, size = 0.7) + #facet_wrap(~thread_id) +
  #geom_point() +
  xlab('Frame (M)') +
  ylab('Extrinsic Reward')
p1

p2 <- ggplot(all_data, aes(x = global_frame, 
                           y = extrinsic.reward, color=thread_id)) +
  geom_smooth(se = FALSE, size = 0.7) +
  labs(color = "Thread ID") +
  xlab('Frame (M)') +
  ylab('Extrinsic Reward')
p2


## CODE FOR EVALUTION
eval_name <- paste('IM-',game,'-trained_eval.csv', sep='')
eval_df <- read.csv(eval_name)
score <- mean(eval_df$reward)
score
