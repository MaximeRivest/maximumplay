library(tidyverse)
library(ggplot2)
library(readr)
library(svglite)
library(ggrepel)
library(ggnewscale)

# Read the main data, skipping the first two lines
data <- read_csv("src/musing_on_tools/multiTimeline (2).csv", skip = 2)

# Read the anthropic mcp data from the separate file
anthropic_mcp_data <- read_csv("src/musing_on_tools/multiTimeline (3).csv", skip = 2)

# Clean up column names for easier use
names(data) <- c("week", "function_calling_openai", "cursor_mcp", "anthropic_computer_use", "gemini_mcp", "openai_mcp")
names(anthropic_mcp_data) <- c("week", "function_calling_openai", "cursor_mcp", "anthropic_computer_use", "anthropic_mcp", "openai_mcp")

# Convert week to Date and ensure data is numeric
data <- data %>%
  mutate(week = as.Date(week)) %>%
  mutate(across(-week, as.numeric))

anthropic_mcp_data <- anthropic_mcp_data %>%
  mutate(week = as.Date(week)) %>%
  mutate(across(-week, as.numeric))

# Add anthropic_mcp column to main data by joining
data <- data %>%
  left_join(anthropic_mcp_data %>% select(week, anthropic_mcp), by = "week")

# Reshape data for plotting with ggplot2
plot_data <- data %>%
  pivot_longer(
    cols = -week,
    names_to = "series",
    values_to = "interest"
  ) %>%
  mutate(series_label = case_when(
    series == "function_calling_openai" ~ "Function Calling (OpenAI)",
    series == "cursor_mcp" ~ "Cursor MCP",
    series == "anthropic_computer_use" ~ "Anthropic Computer Use",
    series == "gemini_mcp" ~ "Gemini MCP",
    series == "anthropic_mcp" ~ "Anthropic MCP",
    series == "openai_mcp" ~ "OpenAI MCP"
  ))

# Define a color palette based on the new specification
color_mapping <- c(
  "Function Calling (OpenAI)" = "#ffffff",  # OpenAI Function Calling: white
  "Cursor MCP" = "#e9e9e9",
  "Anthropic Computer Use" = "#d97757",
  "Gemini MCP" = "#0089f3",
  "Anthropic MCP" = "#ff6b35",  # New orange color for Anthropic MCP
  "OpenAI MCP" = "#ffffff"
)

# Find specific peaks for each series
takeoff_points <- plot_data %>%
  group_by(series_label) %>%
  group_split() %>%
  map_dfr(function(series_data) {
    if (max(series_data$interest, na.rm = TRUE) == 0) return(NULL)
    
    series_name <- series_data$series_label[1]
    
    if (series_name == "Function Calling (OpenAI)") {
      # Keep the existing logic for blue - first significant peak
      threshold <- max(3, 0.1 * max(series_data$interest, na.rm = TRUE))
      first_significant <- series_data %>%
        filter(interest >= threshold) %>%
        slice(1)
      
      if (nrow(first_significant) == 0) return(NULL)
      
      peak_after_start <- series_data %>%
        filter(week >= first_significant$week) %>%
        slice(1:min(20, n())) %>%
        filter(interest == max(interest)) %>%
        slice(1)
      
      return(peak_after_start)
      
    } else if (series_name == "Anthropic Computer Use") {
      # Keep the existing logic for yellow - first significant peak
      threshold <- max(3, 0.1 * max(series_data$interest, na.rm = TRUE))
      first_significant <- series_data %>%
        filter(interest >= threshold) %>%
        slice(1)
      
      if (nrow(first_significant) == 0) return(NULL)
      
      peak_after_start <- series_data %>%
        filter(week >= first_significant$week) %>%
        slice(1:min(10, n())) %>%
        filter(interest == max(interest)) %>%
        slice(1)
      
      return(peak_after_start)
      
    } else if (series_name == "Cursor MCP") {
      # Red marker on 3rd highest peak
      peaks <- series_data %>%
        arrange(desc(interest)) %>%
        slice(1:3)
      
      if (nrow(peaks) >= 3) {
        return(peaks[3,])
      } else {
        return(peaks[nrow(peaks),])
      }
      
    } else if (series_name == "Gemini MCP") {
      # Green marker on 2nd highest peak
      peaks <- series_data %>%
        arrange(desc(interest)) %>%
        slice(1:2)
      
      if (nrow(peaks) >= 2) {
        return(peaks[2,])
      } else {
        return(peaks[nrow(peaks),])
      }
      
    } else if (series_name == "Anthropic MCP") {
      # Orange marker on highest peak
      peaks <- series_data %>%
        arrange(desc(interest)) %>%
        slice(1:1)
      
      return(peaks[1,])
      
    } else if (series_name == "OpenAI MCP") {
      # For purple, find first significant peak like the others
      threshold <- max(3, 0.1 * max(series_data$interest, na.rm = TRUE))
      first_significant <- series_data %>%
        filter(interest >= threshold) %>%
        slice(1)
      
      if (nrow(first_significant) == 0) return(NULL)
      
      peak_after_start <- series_data %>%
        filter(week >= first_significant$week) %>%
        slice(1:min(10, n())) %>%
        filter(interest == max(interest)) %>%
        slice(1)
      
      return(peak_after_start)
    }
    
    return(NULL)
  })

# Define key event dates based on the chapter1.py annotations
event_dates <- tibble(
  date = as.Date(c("2023-06-13", "2024-05-30", "2024-07-23")),
  label = c("OpenAI Function Calling", "Anthropic Tool Use", "Llama 3.1 Tool Use"),
  # Shift label position horizontally so it is always to the left of the dashed line
  x_pos = as.Date(c("2023-06-13", "2024-05-30", "2024-07-23")) - 30,  # 30 days to the left for all
  y_pos = c(85, 95, 90) # Higher y so they sit at the top
)

# Add a color column to takeoff_points for label coloring
plot_color_mapping <- c(
  "Function Calling (OpenAI)" = "#ffffff",
  "Cursor MCP" = "#e9e9e9",
  "Anthropic Computer Use" = "#d97757",
  "Gemini MCP" = "#0089f3",
  "Anthropic MCP" = "#ff6b35",
  "OpenAI MCP" = "#ffffff"
)
takeoff_points <- takeoff_points %>% mutate(label_color = plot_color_mapping[series_label])

# Add a color column to event_dates for label coloring
# (order: OpenAI, Anthropic, Llama)
event_label_colors <- c("#ffffff", "#d97757", "#0064e0")
event_dates$label_color <- event_label_colors

# A custom dark theme for an infographic style
infographic_dark_theme <- function() {
  theme_minimal(base_family = "sans") +
  theme(
    plot.title = element_text(size = 24, face = "bold", hjust = 0.5, color = "#0c85cc", margin = margin(b = 10)),
    plot.subtitle = element_text(size = 18, hjust = 0.5, color = "#0c85cc", margin = margin(b = 25)),
    plot.caption = element_text(size = 12, hjust = 1, color = "#0c85cc", margin = margin(t = 15)),
    axis.title = element_blank(),
    axis.text = element_text(size = 16, color = "#0c85cc"),
    axis.text.x = element_text(margin = margin(t = 5)),
    axis.text.y = element_text(margin = margin(r = 5)),
    legend.position = "none", # We are using direct labels
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.background = element_rect(fill = "#141414", color = NA),
    panel.background = element_rect(fill = "#141414", color = NA),
    plot.margin = margin(20, 40, 20, 20)
  )
}

# Create custom nudge values for better positioning
takeoff_points <- takeoff_points %>%
  mutate(
    nudge_x = case_when(
      series_label == "Function Calling (OpenAI)" ~ -45,  # Further left
      series_label == "Cursor MCP" ~ 40,   # Far right
      series_label == "Anthropic MCP" ~ 60,   # Much further right
      series_label == "OpenAI MCP" ~ 10,   # Slight right
      series_label == "Anthropic Computer Use" ~ -20,  # Left
      series_label == "Gemini MCP" ~ 20,  # Slight right
      TRUE ~ 0
    ),
    nudge_y = case_when(
      series_label == "Function Calling (OpenAI)" ~ 8,
      series_label == "Cursor MCP" ~ 12,
      series_label == "Anthropic MCP" ~ 5,
      series_label == "OpenAI MCP" ~ 15,
      series_label == "Anthropic Computer Use" ~ 7,
      series_label == "Gemini MCP" ~ 7,
      TRUE ~ 7
    )
  )

# Create the plot
p <- ggplot(plot_data, aes(x = week, y = interest, color = series_label, group = series_label)) +
  geom_line(linewidth = 1.5, alpha = 0.9) +
  
  # Add vertical lines for key events, each with a specific color
  geom_vline(
    xintercept = as.numeric(as.Date("2023-06-13")),
    linetype = "dashed", color = "#ffffff", linewidth = 0.8
  ) +
  geom_vline(
    xintercept = as.numeric(as.Date("2024-05-30")),
    linetype = "dashed", color = "#d97757", linewidth = 0.8
  ) +
  geom_vline(
    xintercept = as.numeric(as.Date("2024-07-23")),
    linetype = "dashed", color = "#0064e0", linewidth = 0.8
  ) +
  # Add text labels for the events at the top - individual geom_text for each event
  geom_text(data = event_dates[1,], aes(x = x_pos, y = y_pos, label = label),
            inherit.aes = FALSE,
            hjust = 1, vjust = 0.5, color = "#ffffff", size = 8,
            fontface = "italic", angle = 90) +
  geom_text(data = event_dates[2,], aes(x = x_pos, y = y_pos, label = label),
            inherit.aes = FALSE,
            hjust = 1, vjust = 0.5, color = "#d97757", size = 8,
            fontface = "italic", angle = 90) +
  geom_text(data = event_dates[3,], aes(x = x_pos, y = y_pos, label = label),
            inherit.aes = FALSE,
            hjust = 1, vjust = 0.5, color = "#0064e0", size = 8,
            fontface = "italic", angle = 90) +
  
  # Highlight and annotate the specific points for each series
  geom_point(data = takeoff_points, aes(fill = series_label), color = "white", size = 5, shape = 21, stroke = 1.5, show.legend = FALSE) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "Function Calling (OpenAI)",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#ffffff", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "Function Calling (OpenAI)",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "Function Calling (OpenAI)",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "Cursor MCP",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#e9e9e9", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "Cursor MCP",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "Cursor MCP",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "Anthropic Computer Use",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#d97757", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "Anthropic Computer Use",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "Anthropic Computer Use",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "Gemini MCP",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#0089f3", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "Gemini MCP",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "Gemini MCP",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "Anthropic MCP",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#ff6b35", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "Anthropic MCP",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "Anthropic MCP",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  geom_text_repel(data = takeoff_points[takeoff_points$series_label == "OpenAI MCP",], 
            aes(label = paste(series_label, format(week, "%b %Y"), sep = "\n")),
            color = "#ffffff", fontface = "bold", size = 5,
            box.padding = 2.0, point.padding = 1.5, segment.color = NA,
            min.segment.length = 0, force = 2,
            nudge_x = takeoff_points[takeoff_points$series_label == "OpenAI MCP",]$nudge_x,
            nudge_y = takeoff_points[takeoff_points$series_label == "OpenAI MCP",]$nudge_y,
            hjust = 0.5, lineheight = 1.1) +
  
  # Apply the custom theme and color scale
  infographic_dark_theme() +
  scale_color_manual(values = color_mapping) +
  scale_fill_manual(values = color_mapping) +
  
  # Adjust axis limits to provide space for labels - make it more square
  scale_x_date(breaks = as.Date(c("2023-01-01", "2024-01-01", "2025-01-01")), 
             date_labels = "%Y", expand = expansion(mult = c(0.05, 0.15))) +
  scale_y_continuous(limits = c(0, 110), breaks = seq(0, 100, 25)) +
  
  # Add informative titles and a caption
  labs(
    title = "The Rise of 'Model Context Protocol'",
    subtitle = "Highlighting the take-off point for each trend",
    caption = "Source: Google Trends. Data from 2023-01-01 to 2025-06-15."
  )

# Save the plot in both SVG and PNG formats with square aspect ratio
ggsave("src/musing_on_tools/mcp_trends_infographic.svg", plot = p, width = 12, height = 12)
ggsave("src/musing_on_tools/mcp_trends_infographic.png", plot = p, width = 12, height = 12, dpi = 300)

# Display the plot
print(p)
