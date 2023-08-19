# Program: get-google-trends
# Purpose: get Google trends of benefits extension.
# 
# Date Started: 2020-09-26
# Date Revised: 2023-08-17

library(tidyverse)
library(lubridate)
library(gtrendsR)
library(here)

file_prg <- "07-get-google-trends"

# Golden ratio plotting parameters
mywidth <- 6
golden <- 0.5*(1 + sqrt(5))
myheight <- mywidth / golden

# Google trends -----------------------------------------------------------

# Check out geographic region
# The next line loads the dataset countries
data("countries")
countries <- as_tibble(countries)
names(countries)
countries_US <- countries %>% 
  filter(country_code == "US",
         name == "UNITED STATES")
data("categories")
cats <- as_tibble(categories)

# Set up dates
dat_start_date <- tibble(
  date_start = c("2020-01-01", "2019-01-01"),
)

dat_keyword <- tibble(
  keyword = c(
    "unemployment benefits extension",
    "unemployment extension 2020",
    "stimulus unemployment benefits extension",
    "unemployment benefits $600 extension",
    "600 unemployment benefits",
    "600 unemployment extension",
    "unemployment benefits extension update",
    "stimulus check",
    "update on unemployment extension benefits",
    "second stimulus check",
    "stimulus check 2",
    "extension of unemployment benefits ny",
    "extension of unemployment benefits ca",
    "extension of $600 unemployment benefits",
    "2nd stimulus check",
    "unemployment benefits",
    "federal unemployment extension",
    "extended unemployment"
  )
)

# by = character() performs a cross-join,
# generating all combinations of x and y
dat_get <- full_join(dat_start_date, dat_keyword,
                     by = character()) %>% 
  arrange(keyword, date_start) %>% 
  mutate(date_end = today())

fun_get_trends <- function(pstart, pend, pkeyword) {
  search_window <- paste(pstart, pend)
  
  dat <- gtrends(
    keyword = pkeyword,
    geo = "US",
    time = search_window,
    category = 0,
    gprop = c("web")
  ) 
  
  list(
    interest_over_time = dat$interest_over_time,
    interest_by_country = dat$interest_by_country,
    interest_by_region = dat$interest_by_region,
    interest_by_dma = dat$interest_by_dma,
    interest_by_city = dat$interest_by_city,
    related_topics = dat$related_topics,
    related_queries = dat$related_queries
  )
}

fun_interest <- function(my_list) {
  dat <- as_tibble(my_list$interest_over_time)
  dat <- dat %>% 
    mutate(hits = as.integer(str_replace(hits, "<1", "0.5")))
}

# Get everything 
dat <- dat_get %>%
  mutate(dat_gtrend = pmap(
    list(
      pstart = date_start,
      pend = date_end,
      pkeyword = keyword
    ),
    fun_get_trends
  ))

# Save ALL the data as rds 
fout_rds <- paste0("dat_", file_prg, "-", today(), ".rds")
write_rds(dat, here("out", fout_rds))

# Grab only list element "interest_over_time" from data
dat_hits <- dat %>% 
  mutate(interest = map(dat_gtrend, fun_interest))

# Remove other datasets in list, unnest
dat_hits <- dat_hits %>% 
  select(-dat_gtrend) %>% 
  rename(keyword_check = keyword) %>% 
  unnest(interest)

stopifnot(dat_hits$keyword_check == dat_hits$keyword)

dat_hits <- dat_hits %>% 
  select(-keyword_check)

# Save the file
# Since the search query uses "today", 
# then need to save date to file name
fout <- paste0("dat_", file_prg, "-", today(), ".csv")
write_csv(dat_hits, here("out", fout))