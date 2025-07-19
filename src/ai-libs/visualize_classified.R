#!/usr/bin/env Rscript

# Load libraries
library(ggplot2)
library(dplyr)
library(scales)
library(forcats)
library(tidyr)

# Set working directory
setwd("/home/maxime/Projects/maximumplay/src/ai-libs")

# Read the final classified data
df <- read.csv("pypi_classified_final.csv", stringsAsFactors = FALSE)

# Fix character encoding
df$project <- gsub("‑", "-", df$project)

# Get top 25 by downloads
top_25 <- df %>%
  filter(!is.na(downloads_last_month)) %>%
  top_n(25, downloads_last_month) %>%
  arrange(desc(downloads_last_month)) %>%
  mutate(
    downloads_millions = downloads_last_month / 1e6,
    rank = row_number()
  )

# Define color palette for categories
category_colors <- c(
  "ai_programming" = "#3498db",           # Blue
  "model_provider" = "#e74c3c",           # Red
  "local_inference" = "#2ecc71",          # Green
  "structured_output" = "#f39c12",        # Orange
  "agentic_ai" = "#9b59b6",              # Purple
  "workflow_orchestration" = "#1abc9c",   # Turquoise
  "retrieval_rag" = "#34495e",           # Dark gray
  "memory_state" = "#e67e22",            # Dark orange
  "evaluation_testing" = "#16a085",       # Teal
  "observability_monitoring" = "#8e44ad", # Dark purple
  "prompt_engineering" = "#f1c40f"        # Yellow
)

# Create main plot with color by category
p1 <- ggplot(top_25, aes(x = reorder(project, downloads_last_month), 
                          y = downloads_millions,
                          fill = primary_tag)) +
  geom_bar(stat = "identity", width = 0.8) +
  coord_flip() +
  scale_fill_manual(values = category_colors, name = "Category") +
  scale_y_continuous(
    labels = function(x) paste0(x, "M"),
    breaks = seq(0, 80, 20)
  ) +
  labs(
    title = "Top 25 AI/LLM Libraries by Category",
    subtitle = "PyPI downloads in the last 30 days",
    x = NULL,
    y = "Downloads (Millions)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 11),
    axis.text.x = element_text(size = 11),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_top25_by_category.png", p1, width = 14, height = 10, dpi = 300, bg = "white")

# Create faceted plot by category
# Group categories for better faceting
top_25 <- top_25 %>%
  mutate(
    category_group = case_when(
      primary_tag %in% c("ai_programming", "workflow_orchestration") ~ "Frameworks & Orchestration",
      primary_tag %in% c("model_provider", "local_inference") ~ "Model Access & Inference",
      primary_tag %in% c("structured_output", "prompt_engineering") ~ "Output & Prompts",
      primary_tag %in% c("agentic_ai", "memory_state") ~ "Agents & Memory",
      primary_tag %in% c("retrieval_rag", "evaluation_testing", "observability_monitoring") ~ "RAG & Testing",
      TRUE ~ "Other Tools"
    )
  )

p2 <- ggplot(top_25, aes(x = reorder(project, downloads_last_month), 
                          y = downloads_millions,
                          fill = primary_tag)) +
  geom_bar(stat = "identity", width = 0.8) +
  coord_flip() +
  scale_fill_manual(values = category_colors, name = "Category") +
  scale_y_continuous(
    labels = function(x) paste0(round(x, 1), "M")
  ) +
  facet_wrap(~ category_group, scales = "free", ncol = 2) +
  labs(
    title = "Top 25 AI/LLM Libraries Grouped by Function",
    subtitle = "PyPI downloads in the last 30 days",
    x = NULL,
    y = "Downloads (Millions)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 18, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 9),
    strip.text = element_text(size = 12, face = "bold"),
    strip.background = element_rect(fill = "gray95", color = NA),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_top25_faceted.png", p2, width = 16, height = 12, dpi = 300, bg = "white")

# Create a horizontal grouped bar chart
# Aggregate by category
category_summary <- top_25 %>%
  group_by(primary_tag) %>%
  summarise(
    total_downloads = sum(downloads_millions),
    count = n(),
    avg_downloads = mean(downloads_millions),
    libraries = paste(project, collapse = ", ")
  ) %>%
  arrange(desc(total_downloads))

p3 <- ggplot(category_summary, aes(x = reorder(primary_tag, total_downloads), 
                                    y = total_downloads,
                                    fill = primary_tag)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = paste0(count, " libraries")), 
            hjust = -0.1, size = 4) +
  coord_flip() +
  scale_fill_manual(values = category_colors, guide = "none") +
  scale_y_continuous(
    labels = function(x) paste0(x, "M"),
    expand = c(0, 0),
    limits = c(0, max(category_summary$total_downloads) * 1.2)
  ) +
  labs(
    title = "Total Downloads by Category (Top 25 Libraries)",
    subtitle = "Aggregated downloads across all libraries in each category",
    x = NULL,
    y = "Total Downloads (Millions)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 18, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 11),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("pypi_downloads_by_category.png", p3, width = 12, height = 8, dpi = 300, bg = "white")

# Print summary
cat("\n=== Category Summary (Top 25) ===\n")
for(i in 1:nrow(category_summary)) {
  cat(sprintf("\n%s (%d libraries, %.1fM total downloads):\n  %s\n",
              category_summary$primary_tag[i],
              category_summary$count[i],
              category_summary$total_downloads[i],
              category_summary$libraries[i]))
}

cat("\nVisualizations saved:\n")
cat("- pypi_top25_by_category.png (single chart with colors)\n")
cat("- pypi_top25_faceted.png (grouped by function)\n")
cat("- pypi_downloads_by_category.png (category totals)\n")