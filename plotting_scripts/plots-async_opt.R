library(ggplot2)
library(dplyr)
require(gridExtra)

game <- 'MsPacman'
folders <- c('30-11-17','fixed_epsilon','bigger_batch_size')
legend_names <- c('Original', 'Fixed final epsilon','Fixed final epsilon + 320 batch size')

all_experiments <- data.frame()
# Read in data
for (j in 1:3) {
  path <- paste('/Users/alextsaptsinos/Documents/_Stanford/Courses/_2017Autumn/CS221/project/cs221project/data/async_dql/',game,'/',folders[j],sep='')
  setwd(path)
  all_data <- read.csv("worker_0.csv")
  all_data$thread_id <- 0
  for (i in 1:15){
    name <- paste(paste("worker", i, sep = "_"), "csv", sep = ".")
    thread_data <- read.csv(name)
    thread_data$thread_id <- i
    all_data <- rbind(all_data, thread_data)
  }
  
  all_data <- all_data %>% 
    mutate(thread_id = as.factor(thread_id), experiment = legend_names[j])
  
  all_experiments <- rbind(all_experiments, all_data)

}

# Restrict to first 25M frames
all_experiments = all_experiments[all_experiments$global_frame<25000000,]

# Convert frames to millions
all_experiments$global_frame <- all_experiments$global_frame / 1000000


p1 <- ggplot(all_experiments, aes(x = global_frame, 
                                  y = reward, group = experiment, color = experiment)) +
  geom_smooth(se = TRUE, size = 0.7) + #facet_wrap(~thread_id) +
  #geom_point() +
  xlab('Frame (M)') +
  ylab('Reward') +
  theme(legend.position="bottom", legend.direction="horizontal", 
        legend.title = element_blank())

p1

