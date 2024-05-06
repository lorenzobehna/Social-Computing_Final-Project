# Final Project

## Searching and filtering relevant videos

- **Step one:** Find relevant YouTube videos with the `search()` endpoint
	- Search Term: "artificial intelligence"
	- Perform 2 search API calls for `videoDuration="medium"` and `videoDuration="long"` to filter out YouTube Shorts. (combined cost of $100 + 100 = 200$)
	- Use `relevanceLanguage="en"`to Filter videos by language, only keep english videos
	- Safe videos in a dataframe together with the metadata: title, publish_date, description, video_id, view_count, and comment_count (maybe later we can add video length and potentially other parameters here too?) 
- **Step two:** Filter out unwanted videos
	- Filter out all videos with a `commentCount` that is below a given threshold, let's say 500 comments for now
	- Still manually check if the top 10 videos are english, just to make sure

## Comments