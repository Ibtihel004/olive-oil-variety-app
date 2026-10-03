# Correlation Matrix Plot - Complete Code
# Run this entire script at once

# 1. Load library
library(ggplot2)

# 2. Read the data
df <- read.csv("data_clean.csv")

# 3. Select the 10 model variables
vars <- c("c160", "c161", "c180", "c181", "c182", "c183", 
          "acidite", "peroxyde", "k232", "k270")

# 4. Compute the correlation matrix
cor_matrix <- cor(df[, vars], use = "pairwise.complete.obs")

# 5. Convert to long format for ggplot
melted_cor <- as.data.frame(as.table(cor_matrix))
names(melted_cor) <- c("Var1", "Var2", "value")

# 6. Create the plot
p <- ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 3) +
  scale_fill_gradient2(low = "#3498db", mid = "white", high = "#e74c3c", 
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    legend.position = "right"
  ) +
  labs(title = "Correlation matrix of the ten model variables",
       x = "", y = "") +
  coord_fixed()

# 7. Display and save
print(p)
ggsave("correlation.png", plot = p, width = 8, height = 7, dpi = 300)