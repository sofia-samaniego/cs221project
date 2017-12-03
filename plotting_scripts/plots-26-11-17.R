
library(ggplot2)
library(dplyr)
require(gridExtra)

setwd('/Users/alextsaptsinos/Documents/_Stanford/Courses/_2017Autumn/CS221/project/cs221project/data/aws/SpaceInvaders/26-11-17')

all_data <- read.csv("worker_0.csv")
all_data$thread_id <- 0
for (i in 1:15){
  name <- paste(paste("worker", i, sep = "_"), "csv", sep = ".")
  thread_data <- read.csv(name)
  thread_data$thread_id <- i
  all_data <- rbind(all_data, thread_data)   
}
all_data <- all_data %>% 
  mutate(thread_id = as.factor(thread_id))

p1 <- ggplot(all_data, aes(x = global_frame, 
                           y = reward), color = "deeppink4") +
  geom_smooth(se = FALSE, size = 0.7) 

p2 <- ggplot(all_data, aes(x = global_frame, 
                           y = reward, 
                           group = thread_id, 
                           color = thread_id)) +
  geom_smooth(se = FALSE, size = 0.7) +
  theme(legend.position="bottom", legend.direction="horizontal") + 
  guides(colour = guide_legend(nrow = 2), size = 0.7)

grid.arrange(p1,p2)


## CODE FOR EVALUTION
eval_df <- read.csv('trained_eval.csv')
score <- mean(eval_df$reward)
score
