# dmap.R
#
# Minimal example for reading a DMAP matrix and summarizing top genes.

data <- read.csv("/Volumes/T9/EpiMECoV/data/GSE153667_DMAP_Matrix.csv")
significant <- subset(data, Pr < 0.05)

library(dplyr)
top_genes <- significant %>%
  group_by(GeneID) %>%
  summarise(count=n()) %>%
  arrange(desc(count))

print(head(top_genes, 10))