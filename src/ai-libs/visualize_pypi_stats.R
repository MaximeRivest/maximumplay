#!/usr/bin/env Rscript

# Load libraries
library(ggplot2)
library(dplyr)
library(scales)
library(forcats)

# Set working directory
setwd("/home/maxime/Projects/maximumplay/src/ai-libs")

# Read the data
df <- read.csv("pypi_downloads_ai_only.csv", stringsAsFactors = FALSE)

# Fix character encoding issues
df$project <- gsub("‑", "-", df$project)

# Clean data - remove rows with NA downloads
df_clean <- df %>%
  filter(!is.na(downloads_last_month)) %>%
  mutate(
    downloads_millions = downloads_last_month / 1e6,
    # Create categories based on download ranges
    category = case_when(
      downloads_last_month >= 50e6 ~ "Mega Popular (50M+)",
      downloads_last_month >= 10e6 ~ "Very Popular (10-50M)",
      downloads_last_month >= 1e6 ~ "Popular (1-10M)",
      downloads_last_month >= 100e3 ~ "Moderate (100K-1M)",
      TRUE ~ "Emerging (<100K)"
    )
  )

# Define color palette
colors <- c(
  "Mega Popular (50M+)" = "#e74c3c",
  "Very Popular (10-50M)" = "#f39c12",
  "Popular (1-10M)" = "#27ae60",
  "Moderate (100K-1M)" = "#3498db",
  "Emerging (<100K)" = "#9b59b6"
)

# 1. Top 50 Horizontal Bar Chart - split into 5 plots of 10 each
library(gridExtra)
library(grid)

top_50 <- df_clean %>% 
  top_n(50, downloads_last_month) %>%
  arrange(desc(downloads_last_month)) %>%
  mutate(rank = row_number())

# Create function to generate plot for each group
create_group_plot <- function(data, start_rank, end_rank, color_low, color_mid, color_high) {
  group_data <- data %>% filter(rank >= start_rank & rank <= end_rank)
  
  # Format y-axis labels based on scale
  max_val <- max(group_data$downloads_millions)
  
  p <- ggplot(group_data, aes(x = reorder(project, downloads_last_month), 
                               y = downloads_millions,
                               fill = downloads_millions)) +
    geom_bar(stat = "identity", width = 0.8) +
    coord_flip() +
    scale_fill_gradient2(
      low = color_low,
      mid = color_mid, 
      high = color_high,
      midpoint = median(group_data$downloads_millions),
      guide = "none"
    ) +
    scale_y_continuous(
      labels = function(x) {
        if (max_val > 10) {
          paste0(round(x), "M")
        } else if (max_val > 1) {
          paste0(round(x, 1), "M")
        } else {
          paste0(round(x * 1000), "K")
        }
      },
      limits = c(0, max_val * 1.1),
      expand = c(0, 0)
    ) +
    labs(
      title = paste0("Rank ", start_rank, "-", end_rank),
      x = NULL,
      y = NULL
    ) +
    theme_minimal(base_size = 10) +
    theme(
      plot.title = element_text(size = 12, face = "bold", margin = margin(b = 5)),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 9),
      axis.text.x = element_text(size = 8),
      plot.margin = margin(5, 5, 5, 5)
    )
  
  return(p)
}

# Create 5 plots
p1 <- create_group_plot(top_50, 1, 10, "#3498db", "#f39c12", "#e74c3c")
p2 <- create_group_plot(top_50, 11, 20, "#3498db", "#9b59b6", "#e74c3c")
p3 <- create_group_plot(top_50, 21, 30, "#3498db", "#27ae60", "#f39c12")
p4 <- create_group_plot(top_50, 31, 40, "#2ecc71", "#3498db", "#9b59b6")
p5 <- create_group_plot(top_50, 41, 50, "#27ae60", "#3498db", "#2980b9")

# Arrange in a 3x2 grid (with last cell empty)
p_combined <- grid.arrange(
  p1, p2, p3, p4, p5,
  ncol = 3, nrow = 2,
  top = grid::textGrob("Top 50 AI/LLM Libraries by PyPI Downloads", 
                       gp = grid::gpar(fontsize = 18, fontface = "bold", lineheight = 1.5)),
  bottom = grid::textGrob("Downloads in the last 30 days | Data source: PyPIStats.org", 
                          gp = grid::gpar(fontsize = 9, col = "gray60"))
)

ggsave("pypi_top50_barplot.png", p_combined, width = 18, height = 12, dpi = 300, bg = "white")

# 2. Log-scale scatter plot with categories
p2 <- ggplot(df_clean, aes(x = seq_along(project), 
                            y = downloads_last_month,
                            color = category,
                            size = downloads_last_month)) +
  geom_point(alpha = 0.7) +
  scale_y_log10(
    labels = scales::comma,
    breaks = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8)
  ) +
  scale_color_manual(values = colors) +
  scale_size_continuous(
    range = c(3, 15),
    guide = "none"
  ) +
  labs(
    title = "AI/LLM Library Popularity Distribution",
    subtitle = "Log scale reveals the full spectrum of library adoption",
    x = "Libraries (ranked by downloads)",
    y = "Downloads (log scale)",
    color = "Popularity Tier"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_log_distribution.png", p2, width = 14, height = 8, dpi = 300, bg = "white")

# 3. Category donut chart
category_summary <- df_clean %>%
  group_by(category) %>%
  summarise(
    count = n(),
    total_downloads = sum(downloads_last_month)
  ) %>%
  mutate(
    category = factor(category, 
                     levels = c("Mega Popular (50M+)", "Very Popular (10-50M)", 
                               "Popular (1-10M)", "Moderate (100K-1M)", 
                               "Emerging (<100K)"))
  )

# Calculate percentages
category_summary <- category_summary %>%
  mutate(
    percentage = count / sum(count) * 100,
    label = paste0(count, " libraries\n", round(percentage, 1), "%")
  )

p3 <- ggplot(category_summary, aes(x = 2, y = count, fill = category)) +
  geom_bar(stat = "identity", width = 1, color = "white", size = 2) +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  scale_fill_manual(values = colors) +
  geom_text(aes(label = label), 
            position = position_stack(vjust = 0.5),
            size = 4,
            fontface = "bold",
            color = "white") +
  labs(
    title = "Distribution of Libraries by Popularity Tier",
    subtitle = "Most AI/LLM libraries are still in early adoption phase",
    fill = "Category"
  ) +
  theme_void(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_category_donut.png", p3, width = 10, height = 10, dpi = 300, bg = "white")

# 4. Top 10 with download numbers as labels (lollipop chart)
top_10 <- df_clean %>% 
  top_n(10, downloads_last_month) %>%
  arrange(desc(downloads_last_month))

p4 <- ggplot(top_10, aes(x = reorder(project, downloads_last_month), 
                          y = downloads_millions)) +
  geom_segment(aes(xend = project, yend = 0), 
               size = 8, 
               color = "gray80") +
  geom_point(size = 12, 
             aes(color = downloads_millions)) +
  geom_text(aes(label = paste0(round(downloads_millions, 1), "M")),
            hjust = -0.3,
            size = 5,
            fontface = "bold") +
  coord_flip() +
  scale_color_gradient2(
    low = "#3498db",
    mid = "#f39c12",
    high = "#e74c3c",
    midpoint = 40,
    guide = "none"
  ) +
  scale_y_continuous(
    limits = c(0, max(top_10$downloads_millions) * 1.15),
    expand = c(0, 0)
  ) +
  labs(
    title = "The AI/LLM Library Elite",
    subtitle = "Top 10 libraries dominate the ecosystem",
    x = NULL,
    y = "Downloads (Millions)"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(size = 24, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 16, color = "gray50", margin = margin(b = 20)),
    panel.grid = element_blank(),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    plot.margin = margin(20, 40, 20, 20)
  )

ggsave("pypi_top10_lollipop.png", p4, width = 12, height = 8, dpi = 300, bg = "white")

# 5. Market share visualization - top 10 vs rest
top_10_total <- sum(top_10$downloads_last_month)
rest_total <- sum(df_clean$downloads_last_month) - top_10_total

market_share <- data.frame(
  category = c("Top 10 Libraries", "All Others"),
  downloads = c(top_10_total, rest_total)
) %>%
  mutate(
    percentage = downloads / sum(downloads) * 100,
    label = paste0(round(percentage, 1), "%")
  )

p5 <- ggplot(market_share, aes(x = "", y = downloads, fill = category)) +
  geom_bar(stat = "identity", width = 1) +
  coord_polar("y", start = 0) +
  scale_fill_manual(values = c("Top 10 Libraries" = "#e74c3c", "All Others" = "#95a5a6")) +
  geom_text(aes(label = label), 
            position = position_stack(vjust = 0.5),
            size = 8,
            fontface = "bold",
            color = "white") +
  labs(
    title = "Market Concentration in AI/LLM Libraries",
    subtitle = "Top 10 libraries account for vast majority of downloads",
    fill = NULL
  ) +
  theme_void(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    legend.position = "bottom",
    legend.text = element_text(size = 12),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_market_share.png", p5, width = 8, height = 8, dpi = 300, bg = "white")

# Print summary statistics
cat("\n=== PyPI AI/LLM Libraries Summary ===\n")
cat(sprintf("Total libraries analyzed: %d\n", nrow(df_clean)))
cat(sprintf("Total downloads (last 30 days): %s\n", 
            format(sum(df_clean$downloads_last_month), big.mark = ",")))
cat(sprintf("Average downloads per library: %s\n", 
            format(round(mean(df_clean$downloads_last_month)), big.mark = ",")))
cat(sprintf("Median downloads per library: %s\n", 
            format(round(median(df_clean$downloads_last_month)), big.mark = ",")))

cat("\n=== Top 5 Libraries ===\n")
top_5 <- df_clean %>% top_n(5, downloads_last_month) %>% arrange(desc(downloads_last_month))
for(i in 1:nrow(top_5)) {
  cat(sprintf("%d. %s: %s downloads\n", 
              i, 
              top_5$project[i], 
              format(top_5$downloads_last_month[i], big.mark = ",")))
}

cat("\n=== Category Breakdown ===\n")
for(i in 1:nrow(category_summary)) {
  cat(sprintf("%s: %d libraries (%.1f%%)\n", 
              category_summary$category[i],
              category_summary$count[i],
              category_summary$percentage[i]))
}

cat("\nVisualizations saved as:\n")
cat("- pypi_top50_barplot.png\n")
cat("- pypi_log_distribution.png\n")
cat("- pypi_category_donut.png\n")
cat("- pypi_top10_lollipop.png\n")
cat("- pypi_market_share.png\n")