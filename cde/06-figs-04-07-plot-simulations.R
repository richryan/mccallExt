# Program: plot-simulation.R
# Purpose: Plot simulation results. The simulations compute the average
# experience of a McCall worker who mis-evaluates:
#    1. The probability that benefits are extended
#    2. The length of extension.
# 
# E-mail: RichRyan@csub.edu
# Date Started: 2023-07-19
# Date Revised: 2023-08-09

library(tidyverse)
library(ggthemes)
library(viridis)
library(here)
library(ggrepel)
library(cowplot)

# File information --------------------------------------------------------

file_prg <- "06-figs-04-07-plot-simulations"
file_out <- c("out")

# Parameters for plotting -------------------------------------------------

# Fonts
# See https://cran.rstudio.com/web/packages/showtext/vignettes/introduction.html
library(showtext)
## Loading Google fonts (https://fonts.google.com/)
font_add_google("Fira Sans", "firaSans")

## Automatically use showtext to render text
showtext_auto()

# Golden ratio plotting parameters
mywidth <- 6
golden <- 0.5*(1 + sqrt(5))
myheight <- mywidth / golden

csub_blue <- rgb(0, 53, 148, maxColorValue = 255)
csub_yellow <- rgb(255,210,0, maxColorValue = 255)

# Read in data transfered from AWS EC2 ------------------------------------

dat <- read_csv(here('out', 'dat_04-mccall-uniform-pr-ext_2023-07-19.csv'))
dat_n <- read_csv(here('out', 'dat_05-mccall-uniform-length-ext_2023-07-19.csv'))

# Plot of welfare for probability of extension -------------------------------------------------------------------

xlab_text <- "Difference between perceived probability of extension and
       true probability of extension"

pr_true <- 0.5
ggplot(data = dat) +
  geom_rect(aes(
    xmin = -0.2,
    xmax = 0.0,
    ymin = -Inf,
    ymax = Inf
  ), fill = csub_blue, alpha = .01 / 2) +
  geom_rect(aes(
    xmin = 0.0,
    xmax = 0.2,
    ymin = -Inf,
    ymax = Inf
  ), fill = csub_yellow, alpha = .01 / 2) +  
  geom_hline(yintercept = 0.0, linetype = 'dashed',
             color = 'gray60') +  
  geom_line(mapping = aes(x = probs_perceived - pr_true, y = relative_welfare),
            color = csub_blue, linewidth = 0.9) +
  labs(x = xlab_text, y = "Relative welfare, percent") +
  theme_tufte(base_family = "firaSans") 

figout <- here(file_out, paste0("fig_", file_prg, "-welfare.pdf"))
ggsave(figout, width = mywidth, height = myheight)

# Plot of welfare for length of extension -------------------------------------------------------------------

xlab_text_N <- "Difference between perceived length of extension and
       true length of extension, weeks"

ext_true <- 25
dat_n <- dat_n %>% 
  mutate(rel_n = length_ext_perceived - ext_true)

ggplot(data = dat_n) +
  geom_rect(aes(
    xmin = -10.0,
    xmax = 0.0,
    ymin = -Inf,
    ymax = Inf
  ), fill = csub_blue, alpha = .01 / 2) +
  geom_rect(aes(
    xmin = 0.0,
    xmax = 10.0,
    ymin = -Inf,
    ymax = Inf
  ), fill = csub_yellow, alpha = .01 / 2) +  
  geom_hline(yintercept = 0.0, linetype = 'dashed',
             color = 'gray60') +  
  geom_line(mapping = aes(x = rel_n, y = relative_welfare),
            color = csub_blue, linewidth = 0.9) +
  labs(x = xlab_text_N, y = "Relative welfare, percent") +
  theme_tufte(base_family = "firaSans") 

figout <- here(file_out, paste0("fig_", file_prg, "-welfare-N.pdf"))
ggsave(figout, width = mywidth, height = myheight)

# Plot too optimistic about pr of extension -------------------------------------

# Aggregate statistics
dat_true <- dat %>% 
  filter(probs_perceived == pr_true)
(print(dat_true))

dat <- dat %>% 
  mutate(stopping_time_optim = max(stopping_time * near(probs_perceived, pr_true)),
         stopping_time_diff = stopping_time - stopping_time_optim,
         accepted_wage_optim = max(accepted_wage * near(probs_perceived, pr_true)),
         accepted_wage_diff = (accepted_wage - accepted_wage_optim) / accepted_wage_optim)

prs_relative <- dat$probs_perceived - pr_true
p1 <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  geom_line(mapping = aes(x = prs_relative, y = stopping_time_diff),
            color = csub_blue, linewidth = 0.9) +
  labs(x = "", y = "Periods unemployed\nrelative to optimal") +
  theme_tufte(base_family = "sans")  

p2 <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  geom_line(mapping = aes(x = prs_relative, y = accepted_wage_diff),
            color = csub_blue, linewidth = 0.9) +
  labs(x = xlab_text, y = "Accepted wage\nrelative to optimal, percent") +
  theme_tufte(base_family = "sans")  

plot_grid(p1, p2, ncol = 1, align = "v", axis = "b") 
figout <- here(file_out, paste0("fig_", file_prg, "-statistics.pdf"))
ggsave(figout, width = mywidth, height = myheight)  

p1annotated <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  geom_line(mapping = aes(x = prs_relative, y = stopping_time_diff),
            color = csub_blue, linewidth = 0.9) +
  annotate("rect",
           xmin = 0, xmax = max(prs_relative),
           ymin = 0.0, ymax = max(dat$stopping_time_diff),
           alpha = 0.2) +
  # annotate("text", x = 0.1, y = -0.01, 
  annotate("text", x = 0.1, y = -0.004, 
           label = "Believe an extension is likely\nso spend too long unemployed...") +
  annotate("segment", x = 0.05, xend = 0.14, y = -0.0025, yend = -0.001, linewidth = 0.7) +
  labs(x = "", y = "Periods unemployed\nrelative to optimal") +
  theme_tufte(base_family = "sans")  
(p1annotated)

p2annotated <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  annotate("rect",
           xmin = 0, xmax = max(prs_relative),
           ymin = 0.0, ymax = max(dat$accepted_wage_diff),
           alpha = 0.2) +
  annotate("text", x = 0.1, y = -0.00004,
           label = "...waiting for a high wage draw") +
  geom_line(mapping = aes(x = prs_relative, y = accepted_wage_diff),
            color = csub_blue, linewidth = 0.9) +
  labs(x = xlab_text, y = "Accepted wage\nrelative to optimal, percent") +
  theme_tufte(base_family = "sans")  

plot_grid(p1annotated, p2annotated, ncol = 1, align = "v", axis = "b") 
figout <- here(file_out, paste0("fig_", file_prg, "-statistics-annotated.pdf"))
ggsave(figout, width = mywidth, height = 1.3 * myheight)  

# Plot too pessimistic about pr of extension -------------------------------------

p3annotated <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  geom_line(mapping = aes(x = prs_relative, y = stopping_time_diff),
            color = csub_blue, size = 0.9) +
  annotate("rect",
           xmin = min(prs_relative), xmax = 0,
           ymin = min(dat$stopping_time_diff), ymax = 0.0, 
           alpha = 0.2) +
  # annotate("text", x = 0.1, y = -0.01, 
  annotate("text", x = 0.1, y = -0.004, 
           label = "Believe an extension is unlikely\nso spend too little time unemployed...") +
  annotate("segment", x = 0.05, xend = -0.05, y = -0.0025, yend = -0.001, size = 0.7) +
  labs(x = "", y = "Periods unemployed\nrelative to optimal") +
  theme_tufte(base_family = "sans")  
(p3annotated)

p4annotated <- ggplot(data = dat) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  annotate("rect",
           xmin = min(prs_relative), xmax = 0, 
           ymin = min(dat$accepted_wage_diff), ymax = 0.0, 
           alpha = 0.2) +
  annotate("text", x = 0.1, y = -0.00003,
           label = "...willing to accept job offers\nthat should be rejected") +
  geom_line(mapping = aes(x = prs_relative, y = accepted_wage_diff),
            color = csub_blue, size = 0.9) +
  labs(x = xlab_text, y = "Accepted wage\nrelative to optimal, percent") +
  theme_tufte(base_family = "sans")  
(p4annotated)

plot_grid(p3annotated, p4annotated, ncol = 1, align = "v", axis = "b") 
figout <- here(file_out, paste0("fig_", file_prg, "-statistics-annotated-pessimistic.pdf"))
ggsave(figout, width = mywidth, height = 1.3 * myheight) 

# Plot too pessimistic about length of extension -------------------------------------

dat_n <- dat_n %>% 
  mutate(stopping_time_optim = max(stopping_time * near(length_ext_perceived, ext_true)),
         stopping_time_diff = stopping_time - stopping_time_optim,
         accepted_wage_optim = max(accepted_wage * near(length_ext_perceived, ext_true)),
         accepted_wage_diff = (accepted_wage - accepted_wage_optim) / accepted_wage_optim)

rel_n_min <- min(dat_n$rel_n)
stopping_time_diff_min <- min(dat_n$stopping_time_diff)
p5annotated <- ggplot(data = dat_n) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  geom_line(mapping = aes(x = rel_n, y = stopping_time_diff),
            color = csub_blue, linewidth = 0.9) +
  annotate("rect",
           xmin = rel_n_min, xmax = 0,
           ymin = stopping_time_diff_min, ymax = 0.0, 
           alpha = 0.2) +
  # annotate("text", x = 0.1, y = -0.01, 
  annotate("text", x = 5, y = -0.02,
           label = "Believe extension will be short\nso spend too little time unemployed...") +
  annotate("segment", x = -1, xend = 4, y = -0.0025, yend = -0.015, size = 0.7) +
  labs(x = "", y = "Periods unemployed\nrelative to optimal") +
  theme_tufte(base_family = "firaSans")  
(p5annotated)


accepted_wage_diff_min <- min(dat_n$accepted_wage_diff)
p6annotated <- ggplot(data = dat_n) +
  geom_hline(yintercept = 0.0, linetype = 'dotted') +
  annotate("rect",
           xmin = rel_n_min, xmax = 0, 
           ymin = accepted_wage_diff_min, ymax = 0.0, 
           alpha = 0.2) +
  annotate("text", x = 5, y = -5e-05,
           label = "...willing to accept job offers\nthat should be rejected") +
  geom_line(mapping = aes(x = rel_n, y = accepted_wage_diff),
            color = csub_blue, size = 0.9) +
  labs(x = xlab_text_N, y = "Accepted wage\nrelative to optimal, percent") +
  theme_tufte(base_family = "sans")  
(p6annotated)

plot_grid(p5annotated, p6annotated, ncol = 1, align = "v", axis = "b") 
figout <- here(file_out, paste0("fig_", file_prg, "-statistics-annotated-pessimistic-N.pdf"))
ggsave(figout, width = mywidth, height = 1.3 * myheight) 
