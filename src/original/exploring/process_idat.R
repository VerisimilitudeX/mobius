# process_idat.R
#
# Example script for loading IDAT files with minfi.
# Install needed packages:
# if (!requireNamespace("BiocManager", quietly=TRUE)) {
#     install.packages("BiocManager")
# }
# BiocManager::install(c(
#     "minfi", 
#     "IlluminaHumanMethylation450kmanifest", 
#     "IlluminaHumanMethylation450kanno.ilmn12.hg19"
# ))
#

library(minfi)
library(limma)

baseDir <- "/Volumes/T9/EpiMECoV/data/GSE93266_RAW"

RGset <- read.metharray.exp(base = baseDir)

qcReport(RGset, pdf="QC_Report.pdf")

MSet <- preprocessIllumina(RGset)
GSet <- mapToGenome(MSet)

plotMDS(getM(GSet), top=1000, gene.selection="common")

sampleSheet <- pData(RGset)
# sampleSheet$Phenotype <- c("Control","Patient","Control", ...)

design <- model.matrix(~ sampleSheet$Phenotype)
colnames(design) <- c("Intercept", "Patient")

fit <- lmFit(getBeta(GSet), design)
fit <- eBayes(fit)
top <- topTable(fit, coef="Patient", number=20)
write.csv(top, file="Differentially_Methylated_Probes.csv")

topProbe <- rownames(top)[1]
boxplot(getBeta(GSet)[topProbe,] ~ sampleSheet$Phenotype,
        main=paste("Methylation for Probe", topProbe),
        xlab="Phenotype", ylab="Beta")