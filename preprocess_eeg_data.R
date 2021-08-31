#the data file is too big for github and can be downloaded from google drive: 
# https://drive.google.com/file/d/0B8ua9hwRhsuwXzZJZEY2VDN0Y0k/view?usp=sharing&resourcekey=0--o-g5cDxEs7Q3Ra0lPdKYQ
df <- readRDS(r"(data\eeg_and_emg_data.RDS)")
df2 <- df[,c(4, 5, 6, 7, 8, 9, 10)]
write.csv(x = df2, file="eeg.csv", row.names = F)
#continueing in python