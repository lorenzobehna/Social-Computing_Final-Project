library(tidyverse)
library(magrittr)
library(quanteda)
library(quanteda.textstats)
library(quanteda.textplots)
library(quanteda.textmodels)
library(jsonlite)
library(stopwords)
library(RColorBrewer)

# Read in the preliminarily filtered and cleaned comments
df_comments <- read_csv("~/projects/Social-Computing_Final-Project/data/comments/R/filtered_and_cleaned_comments.csv")

head(df_comments)
summary(df_comments)

# Convert date columns to datetime
df_comments$comment_publish_date <- as.Date(df_comments$comment_publish_date, format="%Y-%m-%d")
df_comments$video_publish_date <- as.Date(df_comments$comment_publish_date, format="%Y-%m-%d")

# Basic mutations and dropping of unneeded columns
df_comments <- df_comments |>
	mutate(comment_text = gsub("\\b[aA]\\. [iI]\\b|\\b[aA] [iI]\\b|\\b[aA].[iI]\\b", "AI", comment_text, ignore.case = TRUE)) |> # Replace 'A I' and 'A. I' with 'AI' 
	mutate(comment_text = gsub("(?i)artificial inteligence", "Artificial Intelligence", comment_text, perl = TRUE)) |> # Correct misspelling and use PERL for case insensitive tag(?i)
	mutate(comment_text = gsub("\\b[0-9]+\\b", "", comment_text)) |>  # remove all standalone numbers
	select(-video_year, -video_month, -video_running_month)


# Visualise comments count per year
# Group the data by year and count the number of comments
comment_count_by_year <- df_comments |> 
	group_by(comment_year) |> 
	summarise(comment_count = n())

# Create a bar chart using ggplot2
ggplot(comment_count_by_year, aes(x = comment_year, y = comment_count)) +
	geom_bar(stat = "identity") +
	labs(x = "Year", y = "Number of Comments", title = "Comments per Year") +
	scale_x_continuous(name = "Year",
					   breaks = 2017:2024, 
					   labels = 2017:2024) +
	theme_minimal()



# Create the main corpus
comments_corpus <- corpus(df_comments, text_field = "comment_text")
# Save the corups
save(comments_corpus, file = "~/projects/Social-Computing_Final-Project/data/comments/R/comments_corpus.RData")
# Load in corpus if needed
load("~/projects/Social-Computing_Final-Project/data/comments/R/comments_corpus.RData")

## Inspect main Corpus
summary(comments_corpus) |> 
	head()
comments_corpus[1:5]

summary(comments_corpus,
		n = ndoc(comments_corpus)) |> 
	head()

# keywords in context analysis
kwic(tokens(comments_corpus), pattern = phrase("A I"), 5) |> head(50)
kwic(tokens(comments_corpus), pattern = phrase("A. I"), 5) |> head(50)
kwic(tokens(comments_corpus), pattern = phrase("A. I."), 5) |> head(50)
kwic(tokens(comments_corpus), pattern = phrase("A I."), 5) |> head(50)
kwic(tokens(comments_corpus), pattern = phrase("gpt"), 5) |> head(50)
kwic(tokens(comments_corpus), pattern = phrase("Large Language Model"), 5) |> head(50)


# Creating token object for main corpus
tokens_comments <- tokens(comments_corpus,
			   what = c("word"),
			   remove_separators = TRUE,
			   include_docvars = TRUE,
			   ngrams = 1L,
			   remove_numbers = FALSE,
			   remove_punct = FALSE,
			   remove_symbols = FALSE,
			   remove_hyphens = FALSE)
tokens_comments |> head()

# Handling colocations: looking at collocations to determine multiwords
coloc6 <- tokens_comments |> 
	tokens_select(pattern = "^[A-Z]",
				  valuetype = "regex",
				  case_insensitive = FALSE,
				  padding = TRUE) |> 
	textstat_collocations(min_count = 5,
						  size = 6,
						  tolower = FALSE)
#coloc6

coloc5 <- tokens_comments |> 
	tokens_select(pattern = "^[A-Z]",
				  valuetype = "regex",
				  case_insensitive = FALSE,
				  padding = TRUE) |> 
	textstat_collocations(min_count = 5,
						  size = 5,
						  tolower = FALSE)
#coloc5

coloc4 <- tokens_comments |> 
	tokens_select(pattern = "^[A-Z]",
				  valuetype = "regex",
				  case_insensitive = FALSE,
				  padding = TRUE) |> 
	textstat_collocations(min_count = 5,
						  size = 4,
						  tolower = FALSE)
#coloc4

coloc3 <- tokens_comments |> 
	tokens_select(pattern = "^[A-Z]",
				  valuetype = "regex",
				  case_insensitive = FALSE,
				  padding = TRUE) |> 
	textstat_collocations(min_count = 5,
						  size = 3,
						  tolower = FALSE)
#coloc3

coloc2 <- tokens_comments |> 
	tokens_select(pattern = "^[A-Z]",
				  valuetype = "regex",
				  case_insensitive = FALSE,
				  padding = TRUE) |> 
	textstat_collocations(min_count = 5,
						  size = 2,
						  tolower = FALSE)
#coloc2


# Creating multiword list
multiword <- c(
	"Chat GPT",
	"Artificial Intelligence",
	"Elon Musk",
	"Machine Learning",
	"Ex Machina",
	"Joe Rogan",
	"John Oliver",
	"I Robot",
	"Turing Test",
	"Star Trek",
	"Neural Network",
	"Reinforcement Learning",
	"Social Media"
)

# Adding the multiwords to the token objects
tokens_comments <- tokens_compound(tokens_comments, pattern = phrase(multiword))

## FURTHER TEXT CLEANING

# remove punctuation 
tokens_comments <- 
	tokens_comments |> 
	tokens_remove(pattern = "^[[:punct:]]+$",
				  valuetype = "regex",
				  padding = TRUE)

# make lower case
tokens_comments <- 
	tokens_comments |> 
	tokens_tolower()

# remove stopwords
tokens_comments <- 
	tokens_comments |> 
	tokens_remove(stopwords("english"), padding = TRUE) 

# stemming 
tokens_comments <- 
	tokens_comments |> 
	tokens_wordstem("english")

# remove empty tokens 
tokens_comments <- 
	tokens_comments |> 
	tokens_remove("") |> 
	tokens_remove(" ")

# creating a lowercase multiword list
multiword_lowercase <- c(
	"chat gpt",
	"artificial intelligence",
	"elon musk",
	"machine learning",
	"ex machina",
	"joe rogan",
	"john oliver",
	"i robot",
	"turing test",
	"star trek",
	"neural network",
	"reinforcement learning",
	"social media"
)

# Adding the lowercase multiwords to the token objects
tokens_comments <- tokens_compound(tokens_comments, pattern = phrase(multiword_lowercase))

# Save the token objects
save(tokens_comments, file = "~/projects/Social-Computing_Final-Project/data/comments/R/toksens_comments.RData")

# If needed load the token objects
load("~/projects/Social-Computing_Final-Project/data/comments/R/toksens_comments.RData")


# Create a document-feature-matrix 
mydfm_tokens <- dfm(tokens_comments)
# Save the token document feature matrix
save(mydfm_tokens, file = "~/projects/Social-Computing_Final-Project/data/comments/R/mydfm_tokens.RData")


## SENTIMENT ANALYSIS

# Matching tokens to sentiment dictionary
tokens_sentiment <- 
	tokens_lookup(
		tokens_comments,
		dictionary = data_dictionary_LSD2015[1:2])
head(tokens_sentiment, 20)

# Creating dfm out of toks_sentiment
dfm_sentiment_entire_timespan <- dfm(tokens_sentiment)

# Converting it back to a data frame with all the docvars
ai_sentiment_entire_timespan <- 
	cbind(convert(dfm_sentiment_entire_timespan, to = "data.frame"), docvars(dfm_sentiment_entire_timespan)) |> 
	mutate(pos_to_neg = positive / (positive + negative))

print(ai_sentiment_entire_timespan)

# Visualising the sentiment over time
dfm_sentiment_graph_entire_timespan <- 
	ai_sentiment_entire_timespan |> 
	ggplot(aes(x = comment_running_month,
			   y = pos_to_neg)) +
	labs(title = "Sentiment across all Comments",
		 subtitle = "Using the LSD2015 Sentiment Dictionary")+
	geom_point(alpha=0.2) +
	scale_x_continuous(name = "Time",
					   breaks = c(seq(1, by = 12, length.out=8)),
					   labels = c(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)) +
	scale_y_continuous(name = "Sentiment (0 = negative, 1 = positive)") +
	theme_minimal()
dfm_sentiment_graph_entire_timespan

# Visualising the sentiment over time by yearly average
dfm_sentiment_graph_entire_timespan_mean <- 
	ai_sentiment_entire_timespan |> 
	mutate(comment_year = as.numeric(comment_year)) |>  # Ensure comment_year is numeric
	group_by(comment_year) |> 
	summarise(mean=mean(pos_to_neg, na.rm = TRUE)) |> 
	ggplot(aes(x = comment_year,
			   y = mean)) +
	labs(title = "Average Sentiment per Year",
		 subtitle = "Using the LSD2015 Sentiment Dictionary")+
	geom_point()+
	geom_line()+
	scale_x_continuous(name = "Year",
					   breaks = 2017:2024,  # Ensure x-axis has breaks for each year
					   labels = 2017:2024) +
	scale_y_continuous(name = "Sentiment (0 = negative, 1 = positive)",
					   limits = c(0,1)) +
	theme_minimal()
dfm_sentiment_graph_entire_timespan_mean



### We also analysed the sentiment with VADER - the results where very similar to LSD2015

install.packages("vader")
library(vader)

# Analyze sentiment using VADER
vader_scores <- vader_df(df_comments$comment_text)

# Calculate mean sentiment per year
mean_vader_sentiment_per_year <- vader_scores %>%
	mutate(comment_year = df_comments$comment_year) %>%
	group_by(comment_year) %>%
	summarise(mean_sentiment = mean(compound, na.rm = TRUE))

# Visualize the results
ggplot(mean_vader_sentiment_per_year, aes(x = comment_year, y = mean_sentiment)) +
	geom_line() +
	geom_point() +
	scale_x_continuous(name = "Year",
					   breaks = 2017:2024,  # Ensure x-axis has breaks for each year
					   labels = 2017:2024) +
	labs(title = "Average Sentiment per Year using VADER",
		 x = "Year", y = "Sentiment") +
	scale_y_continuous(name = "Sentiment (-1 = negative, 1 = positive)",
					   limits = c(-1,1)) +
	theme_minimal()


### CALCULATING MEAN SENTIMENT BEFORE CHATGPT AND AFTER
dfm_sentiment_timeframe_before <- ai_sentiment_entire_timespan |> 
	filter(comment_running_month >= 70)

dfm_sentiment_timeframe_after <- ai_sentiment_entire_timespan |> 
	filter(comment_running_month < 70)

# Calculate the mean value of pos_to_neg for each timeframe
mean_pos_to_neg_timeframe_before <- dfm_sentiment_timeframe_before |> 
	summarise(mean_pos_to_neg = mean(pos_to_neg, na.rm = TRUE))

mean_pos_to_neg_timeframe_after <- dfm_sentiment_timeframe_after |> 
	summarise(mean_pos_to_neg = mean(pos_to_neg, na.rm = TRUE))

# Print the results
print(paste("Mean pos_to_neg for timeframe before:", mean_pos_to_neg_timeframe_before$mean_pos_to_neg))
print(paste("Mean pos_to_neg for timeframe after:", mean_pos_to_neg_timeframe_after$mean_pos_to_neg))

## TOPIC MODELLING 
library(topicmodels)
library(tm)

# Creating document term matrix of pre and post twitter data 
#trimming
dtm_comments <- mydfm_tokens |> 
	dfm_trim(sparsity = 0.999) |> 
	convert(to = "topicmodels")

as.matrix(dtm_comments)[1:10,1:10]
# remove empty rows
dtm_comments <- dtm_comments[rowSums(as.matrix(dtm_comments)) > 0, ]

# create a topic model pre twitter
lda_model_comments <- LDA(dtm_comments, k = 5)

terms(lda_model_comments)

# pull out posterior distribution
posterior_distribution <- posterior(lda_model_comments)
posterior_distribution_term_over_topic <- posterior_distribution$terms
posterior_distribution_topic_over_documents <- posterior_distribution$documents

# pull out top topic per document pre twitter
top_topic_per_document <- 
	topics(lda_model_comments)
top_topic_per_document |> head()

# pull out top term per topic pre twitter
top_terms_per_topic <-
	terms(lda_model_comments, 20)
top_terms_per_topic

library(wordcloud)
# helper function
wordcloud_topic <- function(topics, i) { 
	cloud.data <- sort(topics[i, ], decreasing = T)[1:100]
	wordcloud(names(cloud.data), 
			  freq = cloud.data, 
			  scale = c(4, 0.4), 
			  min.freq = 1, 
			  rot.per = 0, 
			  random.order = F) 
}

# Set the working directory to where you want to save the files
setwd("~/projects/Social-Computing_Final-Project/data/comments/R/")

# Function to generate and save word clouds for each topic
save_wordcloud <- function(topic_number) {
	png_filename <- paste0("wordcloud_top_", topic_number, ".png")
	png(png_filename)
	wordcloud_topic(posterior_distribution_term_over_topic, topic_number)
	dev.off()
}

# Generating and saving word clouds for topics 1, 2, and 3
save_wordcloud(1)
save_wordcloud(2)
save_wordcloud(3)

## STRUCTURAL TOPIC MODELL
library(stm)
# creating a document term matrix for the structural topic model
dtm_stm <- mydfm_tokens |> 
	dfm_trim(sparsity = 0.999) |> 
	convert(to = "stm")

# number of topics
k = 5

# creating the structural topic model
stm_model <- stm(dtm_stm$documents,
				 dtm_stm$vocab,
				 K = k,
				 data = NULL,
				 init.type = "LDA")
# label the topics
labelTopics(stm_model)

# create a second structural topic model over time using comment_running_month
stm_model_2 <- stm(dtm_stm$documents,
				   dtm_stm$vocab,
				   K = k,
				   prevalence = ~ s(comment_running_month),
				   data = select(dtm_stm$meta, comment_running_month),
				   max.em.its = 5000,
				   init.type = "LDA")

labelTopics(stm_model_2, n = 5)
sl <- sageLabels(stm_model_2, n = 5)

highest_probability <- sl$marginal$prob

frex <- sl$marginal$frex
l_f1 <- paste0(frex[1,],collapse = " ")
l_f2 <- paste0(frex[2,],collapse = " ")
l_f3 <- paste0(frex[3,],collapse = " ")
l_f4 <- paste0(frex[4,],collapse = " ")
l_f5 <- paste0(frex[5,],collapse = " ")

l_f <- c(l_f1, l_f2, l_f3, l_f4, l_f5)

predicted_probability <- estimateEffect(1:k ~ s(comment_running_month),
										stm_model_2,
										meta = select(dtm_stm$meta, comment_running_month),
										uncertainty = "Global")

# visualise and print the stm
#png("stm_results.png", height = 800, width = 700)
par(mfrow = c(5, 1), mar = c(3, 4, 1, .5))
for (i in 1:k) {
	plot.estimateEffect(predicted_probability, 
						"comment_running_month", 
						method = "continuous", 
						topics = i, 
						model = z, 
						xaxt = "n", 
						printlegend = FALSE, 
						xlab = "comment_running_month", 
						# ylim = c(0, 0.160),
						main = paste0("TOPIC ",i,": ",l_f[i] ))
	axis(1, 
		 at = seq(1,by = 12, length.out = 8),
		 labels = c(2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024))
}
#dev.off()
