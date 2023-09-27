# Program: 02-plot-sequence-reservation-wages.R
# Purpose: Plot sequences of reservation wages.
# 
# Date Started: 2023-07-18
# Date Revised: 2023-07-18
library(tidyverse)
library(tidyquant)
library(ggthemes)
library(viridis)
library(ggrepel)
library(here)
library(extrafont)

# Add fonts
# See https://cran.rstudio.com/web/packages/showtext/vignettes/introduction.html
library(showtext)
## Loading Google fonts (https://fonts.google.com/)
font_add_google("Fira Sans", "firaSans")

## Automatically use showtext to render text
showtext_auto()

file_prg <- "02-fig-03-plot-sequence-reservation-wages"
input_file <- here("out", "dat_01-get-sequence-reservation-wages-seq-reservation-wages.csv")
input_file_prop <- here("out", "dat_01-get-sequence-reservation-wages-seq-reservation-wages-properties.csv")
file_out <- c("out")

csub_blue <- rgb(0, 26, 112, maxColorValue = 255)
csub_gray <- rgb(112, 115, 114, maxColorValue = 255)

# Golden ratio plotting parameters
mywidth <- 6
golden <- 0.5*(1 + sqrt(5))
myheight <- mywidth / golden

# Read in datasets ---------------------------------------------------------

dat <- read_csv(input_file)
dat_prop <- read_csv(input_file_prop)

# Plot sequences of reservation wages -------------------------------------

ggplot(data = dat) +
  geom_line(mapping = aes(x = horizon, y =  reservation_wages, color = pr_ext))

dat_plt <- dat %>% 
  mutate(series = case_when(
    pr_ext == "extended" ~ "Extended",
    TRUE ~ paste0(round(as.numeric(pr_ext), digits = 1))
  ),
  series = str_remove(series, "^0+"),
  series_lbl = case_when(
    series == "Extended" & horizon == 1  ~ "Extended",
    series == ".1"       & horizon == 2  ~ "Probability of extension 0.1",
    series == ".5"       & horizon == 8  ~ "Probability of extension 0.5",
    series == ".9"       & horizon == 15 ~ "Probability of extension 0.9",    
    TRUE ~ ""
      )) %>% 
  filter(series %in% c("Extended", ".1", ".5", ".9")) %>% 
  mutate(series = factor(series, levels = c("Extended", ".1", ".5", ".9")))
  

periods_ui_compensation_max <- max(dat_plt$horizon)
brks <- seq(from = 0, to = periods_ui_compensation_max, by = 1)
brks_lbl <- case_when(brks %% 5 == 0 ~ paste0(brks),
                      TRUE ~ "")
ggplot(dat = dat_plt) +
  geom_line(mapping = aes(x = horizon, y =  reservation_wages, color = series, linetype = series), 
            linewidth = 1.0) +
  geom_text_repel(mapping = aes(x = horizon, y = reservation_wages, label = series_lbl, color = series),
                  max.overlaps = Inf,
                  point.padding = 0.3,
                  box.padding = 2.0,
                  nudge_x = 10,
                  arrow = arrow(length = unit(0.015, "npc"))) +
  labs(x = "Remaining periods of UI compensation", y = "Reservation wages") +
  scale_color_manual(values = c(csub_blue, "#E0218A", "#ed5c9b", "#f18dbc")) +
  scale_linetype_manual(values = c("solid", "longdash", "dotdash", "dotted")) +
  scale_x_continuous(breaks = brks,
                     labels = brks_lbl) +
  guides(color = "none", linetype = "none") +
  theme_tufte(base_family = 'firaSans', ticks = TRUE)

fout_name <- paste0("fig_", file_prg, ".pdf")
fout <- here(file_out, fout_name)
ggsave(fout, plot = last_plot(), 
       width = mywidth, height = myheight)

# Plot properties of sequences --------------------------------------------

dat_prop_plt <- dat_prop %>% 
  mutate(case = case_when(
    bbeta == max(bbeta) ~ "high_beta",
    flow_nonwork == min(flow_nonwork) ~ "low_z",
    ui_compensation == min(ui_compensation) ~ "low_c"
  )) %>% 
  mutate(series = case_when(
    pr_ext == "extended" ~ "Extended",
    TRUE ~ paste0(round(as.numeric(pr_ext), digits = 1))
  ),
  series = str_remove(series, "^0+"),
  series_lbl = case_when(
    series == "Extended" & horizon == 1  ~ "Extended",
    series == ".1"       & horizon == 2  ~ "Probability of extension 0.1",
    series == ".5"       & horizon == 8  ~ "Probability of extension 0.5",
    series == ".9"       & horizon == 15 ~ "Probability of extension 0.9",    
    .default = ""
      )) %>% 
  filter(series %in% c("Extended", ".1", ".5", ".9")) %>% 
  mutate(series = factor(series, levels = c("Extended", ".1", ".5", ".9"))) %>% 
  bind_rows(mutate(dat_plt, case = "baseline"))

dat_prop_plt <- dat_prop_plt %>% 
  filter(series == "Extended" | series == ".5") %>% 
  mutate(series = case_when(
    series == ".5" ~ "0.5",
    series == "Extended" ~ "Extended"
  ))
ggplot(data = dat_prop_plt) +
  geom_line(mapping = aes(x = horizon, y =  reservation_wages, color = case, linetype = series), linewidth = 1.2) +
  scale_color_viridis_d(labels = c("baseline" = "Baseline", "high_beta" = expression(High~beta), low_c = expression(Low~c), low_z = expression(Low~z)), end = 0.8) +  
  labs(x = "Remaining periods of UI compensation", y = "Reservation wages",
       color = "", linetype = "Extended or\npr. of extension") +
  scale_x_continuous(breaks = brks,
                     labels = brks_lbl) +
  theme_tufte(base_family = 'firaSans', ticks = TRUE) 

fout_properties_name <- paste0("fig_", file_prg, "-properties.pdf")
fout_properties <- here(file_out, fout_properties_name)
ggsave(fout_properties, plot = last_plot(), 
       width = mywidth, height = myheight)
