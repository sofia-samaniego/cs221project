library(ggplot2)
library(dplyr)
require(gridExtra)

game <- 'MsPacman-v0'
folders <- c('1.0','1.0_slow_decay','1.0_no_decay2')
legend_names <- c('Original', 'Slower decay','Natural decay')

all_experiments <- data.frame()
# Read in data
for (j in 1:3) {
  path <- paste('/Users/alextsaptsinos/Documents/_Stanford/Courses/_2017Autumn/CS221/project/cs221project/data/aim_dql/',game,'/',folders[j],sep='')
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
    mutate(thread_id = as.factor(thread_id), experiment = legend_names[j])
  
  all_experiments <- rbind(all_experiments, all_data)
  
}

# Convert frames to millions
all_experiments$global_frame <- all_experiments$global_frame / 1000000


p1 <- ggplot(all_experiments, aes(x = global_frame, 
                                  y = extrinsic.reward, group = experiment, color = experiment)) +
  geom_smooth(se = TRUE, size = 0.7, method="loess") + #facet_wrap(~thread_id) +
  #geom_point() +
  xlab('Frame (M)') +
  ylab('Extrinsic Reward') +
  theme(legend.position="bottom", legend.direction="horizontal", 
        legend.title = element_blank())
p1

p2 <- ggplot(all_experiments, aes(x = global_frame, 
                                  y = intrinsic.reward, group = experiment, color = experiment)) +
  geom_smooth(se = TRUE, size = 0.7, method="loess") + #facet_wrap(~thread_id) +
  #geom_point() +
  xlab('Frame (M)') +
  ylab('Intrinsic Reward') +
  theme(legend.position="bottom", legend.direction="horizontal", 
        legend.title = element_blank())
p2
