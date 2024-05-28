df <- read.csv("/Users/lorenzobehna/projects/Social-Computing_Final-Project/data/comments/R/data.csv", 
			   header = TRUE,
			   na.strings=c("","NA"),    
			   sep = ",")

head(df)

summary(df)

# Convert to date only
df$comment_publish_date <- as.Date(df$comment_publish_date, format="%Y-%m-%d")

type_of(df$video_year)

summary(df)
