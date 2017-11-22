# This script will plot the computation time against number of threads

###########
# CHANGE WD
setwd('/Users/alextsaptsinos/Documents/_Stanford/Courses/_2017Autumn/CS221/project/cs221project/plotting_scripts')
############

library(ggplot2)

thread_df <- read.csv('../data/thread_cpu_data.csv')

myplot <- ggplot(thread_df, aes(x=num_threads, y=time)) + 
  geom_point() + coord_cartesian(xlim=c(0,35), ylim=c(0,360)) +
  xlab('Number of Threads') + ylab('Time (s)')
myplot
ggsave(filename = "thread_plot.png", plot=myplot)
