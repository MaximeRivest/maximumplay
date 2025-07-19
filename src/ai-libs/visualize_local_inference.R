#!/usr/bin/env Rscript

library(ggplot2)
library(dplyr)
library(scales)

# Read the final data
df <- read.csv("pypi_classified_final.csv", stringsAsFactors = FALSE)

# Get all local inference libraries
local_inf <- df %>%
  filter(primary_tag == "local_inference") %>%
  mutate(downloads_millions = downloads_last_month / 1e6)

# Create a focused chart for local inference
p <- ggplot(local_inf, aes(x = reorder(project, downloads_last_month), 
                           y = downloads_last_month,
                           fill = downloads_last_month)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = paste0(round(downloads_last_month/1000), "K")), 
            hjust = -0.1, size = 4) +
  coord_flip() +
  scale_fill_gradient(low = "#27ae60", high = "#16a085", guide = "none") +
  scale_y_continuous(
    labels = function(x) paste0(x/1000, "K"),
    expand = c(0, 0),
    limits = c(0, max(local_inf$downloads_last_month) * 1.2)
  ) +
  labs(
    title = "Local LLM Inference Engines",
    subtitle = "PyPI downloads in the last 30 days",
    x = NULL,
    y = "Downloads",
    caption = paste0("vLLM and SGLang are high-performance serving engines\n",
                     "Ollama and LM Studio provide user-friendly interfaces\n",
                     "MAX (Modular) is the newest entrant with growing adoption")
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(size = 20, face = "bold", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 14, color = "gray50", margin = margin(b = 20)),
    plot.caption = element_text(size = 11, color = "gray60", hjust = 0),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.text.x = element_text(size = 11),
    plot.margin = margin(20, 20, 20, 20)
  )

ggsave("local_inference_engines.png", p, width = 10, height = 6, dpi = 300, bg = "white")

# Print details
cat("\n=== Local Inference Engines ===\n")
for(i in 1:nrow(local_inf)) {
  cat(sprintf("%s (%s): %s downloads\n", 
              local_inf$project[i],
              local_inf$pypi_package[i],
              format(local_inf$downloads_last_month[i], big.mark = ",")))
}

cat("\nVisualization saved: local_inference_engines.png\n")