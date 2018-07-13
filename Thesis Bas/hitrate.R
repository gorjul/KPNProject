#install.packages("dplyr")
install.packages("stringr")
library(dplyr)
library(stringr)



setwd("C:/Users/haver507/Desktop/Thesis")

CLM <- read.csv("Mijn CLM.csv",header=TRUE, sep = ";", stringsAsFactors=FALSE)
HPS <- read.csv("HPS2.csv", header = TRUE, sep = ";", stringsAsFactors=FALSE)
Comp <- read.csv("Companies.csv", header = TRUE, sep = ";",stringsAsFactors=FALSE)

CLM_all <- data.frame(CLM$Sub_BusinessPartner,CLM$Hoofd_BusinessPartner,CLM$Klantnaam,CLM$Account,CLM$CKR_nummer)

CLM1 <- tolower(as.character(CLM$Account)) 
CLM1 <- str_trim(CLM1,"right")

CLM2 <- tolower(as.character(CLM$Klantnaam))
CLM2 <- str_trim(CLM2,"right")


CLM3 <- tolower(as.character(CLM$Sub_BusinessPartner))
CLM3 <- str_trim(CLM3,"right")

CLM4 <- tolower(as.character(CLM$Hoofd_BusinessPartner))
CLM4 <- str_trim(CLM4,"right")

HPS_all <- data.frame(HPS$accountnaam, HPS$VDDSGG_Instellingsnaam, HPS$ckr_nummer_hoofd)

HPS1 <- tolower(as.character(HPS$accountnaam))
HPS1 <- str_trim(HPS1,"right")

HPS2 <-  tolower(as.character(HPS$VDDSGG_Instellingsnaam))
HPS2 <-  str_trim(HPS2,"right")



COMP <- tolower(as.character(Comp$unique.ICT_vac.Company.))

names(CLM1)[1] <- "company"
names(CLM2)[1] <- "company"
names(CLM3)[1] <- "company"
names(CLM4)[1] <- "company"

names(HPS1)[1] <- "company"
names(HPS2)[1] <- "company"


KPN_COMP <- data.frame(rbind(c(CLM1,CLM2,CLM3,CLM4,HPS1,HPS2)))

x <- ACC[1]
print(x)

y <- Comp$unique.ICT_vac.Company.[1]
print(y)


count1 <- data.frame(HPS1[HPS1 %in% COMP])
count2 <- data.frame(HPS2[HPS2 %in% COMP])
names(count1)[1] <- "Company"
names(count2)[1] <- "Company"

count_total <- unique(rbind(count1,count2))
44/1960
