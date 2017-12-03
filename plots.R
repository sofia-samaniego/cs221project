
library(ggplot2)
library(dplyr)
require(gridExtra)

#setwd('/Users/Sofia/Desktop/Stanford/Autumn\ 2017/CS221/project/AWS-data/data/SpaceInvaders/15_11_17')
#num_workers <- 16

setwd('/Users/Sofia/Desktop/Stanford/Autumn\ 2017/CS221/project/cs221project/data')
num_workers <- 4

all_data <- read.csv("worker_0.csv")
all_data$thread_id <- 0
for (i in 1:(num_workers-1)){
    name <- paste(paste("worker", i, sep = "_"), "csv", sep = ".")
    thread_data <- read.csv(name)
    thread_data$thread_id <- i
    all_data <- rbind(all_data, thread_data)   
}
all_data <- all_data %>% 
    mutate(thread_id = as.factor(thread_id))

p1 <- ggplot(all_data, aes(x = global_frame, 
                           y = reward), color = "deeppink4") +
    geom_smooth(se = FALSE, size = 0.7)  + 
    ylab("Reward") + 
    xlab("Frame") 

p2 <- ggplot(all_data, aes(x = global_frame, 
                     y = reward, 
                     group = thread_id, 
                     color = thread_id)) +
    geom_smooth(se = FALSE, size = 0.7) +
    theme(legend.position="bottom", legend.direction="horizontal") + 
    guides(colour = guide_legend(nrow = 2), size = 0.7) +
    ylab("Reward") + 
    xlab("Frame")
    
p3 <- ggplot(all_data, aes(x = global_frame, 
                               y = reward, 
                               group = final_epsilon, 
                               color = final_epsilon)) +
        geom_smooth(se = FALSE, size = 0.7) +
        theme(legend.position="bottom", legend.direction="horizontal") + 
        guides(colour = guide_legend(nrow = 1), size = 0.7) + 
        ylab("Reward") + 
        xlab("Frame")

grid.arrange(p1,p2)

epsilons <- all_data %>% group_by(thread_id) %>% summarise(final_epsilon = as.factor(round(min(epsilon),4)))

all_data <- all_data %>% group_by(thread_id) %>% mutate(final_epsilon = as.factor(round(min(epsilon),4)))

p3 <- ggplot(all_data, aes(x = global_frame, 
                           y = reward, 
                           group = final_epsilon, 
                           color = final_epsilon)) +
    geom_smooth(se = FALSE, size = 0.7) +
    theme(legend.position="bottom", legend.direction="horizontal") + 
    guides(colour = guide_legend(nrow = 1), size = 0.7) + 
    ylab("Reward") + 
    xlab("Frame")

p4 <- ggplot(all_data, aes(x = global_frame, 
                           y = reward, 
                           group = thread_id, 
                           color = final_epsilon)) +
    geom_smooth(se = FALSE, size = 0.7) +
    theme(legend.position="bottom", legend.direction="horizontal") + 
    guides(colour = guide_legend(nrow = 2), size = 0.7) +
    ylab("Reward") + 
    xlab("Frame")

grid.arrange(p1,p4)

p1
