# Optional R template for DCA plotting from exported points
suppressPackageStartupMessages({
  library(readr)
  library(ggplot2)
})

m <- read_csv(file.path("output","intermediate","DCA_points_model_AD.csv"), show_col_types = FALSE)
a <- read_csv(file.path("output","intermediate","DCA_points_treatall_AD.csv"), show_col_types = FALSE)
n <- read_csv(file.path("output","intermediate","DCA_points_treatnone_AD.csv"), show_col_types = FALSE)

m$curve <- "Model"; a$curve <- "Treat all"; n$curve <- "Treat none"
d <- rbind(m,a,n)

p <- ggplot(d, aes(threshold, net_benefit, linetype = curve)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Decision curve analysis", x = "Threshold probability", y = "Net benefit") +
  theme_minimal(base_size = 11)

ggsave(file.path("output","figures","Decision_R.png"), p, width = 6, height = 4.5, dpi = 300)
