# Program: plot-hits.R
# Purpose: Plot number of hits over time.
# 
# Note: To available fonts, use the
# command > windowsFonts(). 
# 
# Date Started: 2020-09-27
# Date Revised: 2023-08-17

library(tidyverse)
library(tidyquant)
library(ggthemes)
library(viridis)
library(gtrendsR)
library(here)
library(ggrepel)

file_prg <- "08-fig-01-plot-google-searches"

# Golden ratio plotting parameters
mywidth <- 6
golden <- 0.5*(1 + sqrt(5))
myheight <- mywidth / golden

# csub_blue <- "slateblue"
csub_blue <- rgb(0, 53, 148, maxColorValue = 255)
csub_yellow <- rgb(255,210,0, maxColorValue = 255)

# Fonts
# See https://cran.rstudio.com/web/packages/showtext/vignettes/introduction.html
library(showtext)
## Loading Google fonts (https://fonts.google.com/)
font_add_google("Fira Sans", "firaSans")

## Automatically use showtext to render text
showtext_auto()

# Read in data ------------------------------------------------------------

fin <- here("out", "dat_07-get-google-trends-2023-08-19.csv")
dat <- read_csv(file = fin)

# dat <- dat %>% 
#   group_by(date_start, date_end, keyword) 

dat_2019 <- dat %>% 
  filter(year(date_start) == 2019) %>% 
  arrange(keyword, date)

dat_2020 <- dat %>% 
  filter(year(date_start) == 2020) %>% 
  arrange(keyword, date)

my_keyword <- "unemployment benefits extension"
my_keyword_name <- str_replace_all(my_keyword, " ", "-")
dat_plt <- dat_2020 %>% 
  filter(keyword == my_keyword) %>% 
  rename(date_time = date) %>% 
  # Create weekly date, datew
  mutate(date = date(date_time)) 

fun_day2wk <- function(vdate_wk, mydate) {
  vwk_int <- interval(vdate_wk, vdate_wk + days(6))
  wk_mydate <- int_overlaps(vwk_int, interval(mydate, mydate))
  # Return week of mydate
  vdate_wk[wk_mydate]
}

# Federal Pandemic Unemployment Compensation program; ie,
# $600 checks run out
day_CARES_pass <- ymd("2020-03-27")
day_600fpuc_expire <- ymd("2020-07-25")
# Or:
# day_600fpuc_expire <- ymd("2020-07-31")
day_peuc_ends <- ymd("2021-09-04")
day_consolidated_appropriations_act <- ymd("2020-12-27")
day_300lwa <- ymd("2020-08-08")
day_continued_assistance_ends <- ymd("2021-03-12")

dat_plt_label <- tribble(
  ~date, ~my_label, ~y,
  fun_day2wk(dat_plt$date, day_CARES_pass), "CARES Act passes", 50,
  fun_day2wk(dat_plt$date, day_peuc_ends), "Temporary UI programs end", 90,
  fun_day2wk(dat_plt$date, day_consolidated_appropriations_act), "Continued Assistance Act signed into law", 90,
  fun_day2wk(dat_plt$date, day_600fpuc_expire), "$600 FPUC expires", 70,
  fun_day2wk(dat_plt$date, day_300lwa), "$300 Lost Wages Assistance signed", 80,
  fun_day2wk(dat_plt$date, day_continued_assistance_ends), "Continued Assistance Act ends;\nAmerican Rescue Plan Act authorizes 29 additional weeks of PEUC benefits", 75
) %>% 
  arrange(date)

# Get the x-axis ticks
chart_early <- ymd("2020-01-01")
chart_late <- ymd("2022-03-01")
my_xticks_major <- seq(chart_early, chart_late, by = "1 months")

fun_make_date_labels <- function(dates) {
  date_labels <- case_when(
    month(dates, label = TRUE) == "Jan" ~ paste(month(dates, label = TRUE), year(dates)),
    month(dates, label = TRUE) == "Apr" ~ paste(month(dates, label = TRUE), year(dates)),
    month(dates, label = TRUE) == "Jul" ~ paste(month(dates, label = TRUE), year(dates)),    
    month(dates, label = TRUE) == "Oct" ~ paste(month(dates, label = TRUE), year(dates)),        
    TRUE ~ "")
  return(date_labels)
}

my_vlinetype <- "longdash"
ggplot(data = filter(dat_plt, date >= chart_early & date <= chart_late)) +
  geom_line(mapping = aes(x = date, y = hits), color = csub_blue, linewidth = 1.5) +
  labs(x = "", y = "Index", ) +
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_CARES_pass), 
             color = "black", linetype = my_vlinetype) +
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_600fpuc_expire), 
             color = "black", linetype = my_vlinetype) +
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_300lwa), 
             color = "black", linetype = my_vlinetype) +  
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_continued_assistance_ends), 
             color = "black", linetype = my_vlinetype) +    
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_peuc_ends), 
             color = "black", linetype = my_vlinetype) +  
  geom_vline(xintercept = fun_day2wk(dat_plt$date, day_consolidated_appropriations_act), 
             color = "black", linetype = my_vlinetype) +    
  scale_x_date(breaks = my_xticks_major,
               labels = fun_make_date_labels(my_xticks_major)) +
  geom_text_repel(data = dat_plt_label,
                  color = "black",
                  size = 6.0,
                  fontface = "bold",
                  mapping = aes(x = date, y = y, label = my_label),
                  box.padding = 1.5,
                  max.overlaps = Inf,
                  # Individually shift labels
                  nudge_x = c(-50, -150, 100, 250, 50, 50),
                  arrow = arrow(length = unit(0.015, "npc"))) +  
  theme_tufte(base_family = 'firaSans', ticks = TRUE) +
  theme(axis.text=element_text(size = 16),
        axis.title=element_text(size = 20, face = "plain"))

        

plt_out <- here("out", paste0("fig_", file_prg, "-", my_keyword_name))

# See: https://stackoverflow.com/questions/14942681/change-size-of-axes-title-and-labels-in-ggplot2
www <- 1920
hhh <- 1051
ggsave(paste0(plt_out, ".png"),
       width = www / 90,
       height = hhh / 90,
       dpi = 900)
ggsave(paste0(plt_out, ".pdf"),
       width = www / 90,
       height = hhh / 90)
