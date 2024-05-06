from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pandas as pd

# open the text file containing the API key
with open("code/YouTube_Data_API_Key.txt", "r") as file:
    api_key = file.read().strip() 

# print("API Key:", api_key)

MAX_RESULTS = 10

def search_videos(query, max_results=MAX_RESULTS, published_after=None, published_before=None):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # convert datetime objects to ISO 8601 string format
    published_after_string = published_after.strftime('%Y-%m-%dT%H:%M:%SZ') if published_after else None
    published_before_string = published_before.strftime('%Y-%m-%dT%H:%M:%SZ') if published_before else None

    search_request = youtube.search().list(q=query, part="snippet", type="video", maxResults=max_results, publishedAfter=published_after_string, publishedBefore=published_before_string)
    search_response = search_request.execute()

    video_ids = [item['id']['videoId'] for item in search_response['items']]

    video_request = youtube.videos().list(part="snippet,statistics", id=",".join(video_ids))
    video_response = video_request.execute()

    videos_data = []
    for item in video_response['items']:
        videos_data.append({
            'title': item['snippet']['title'],
            'publish_date': item['snippet']['publishedAt'],
            'description': item['snippet']['description'],
            'video_id': item['id'],
            'view_count': int(item['statistics']['viewCount']) if 'viewCount' in item['statistics'] else 0
        })

    return pd.DataFrame(videos_data)

# Example usage: Search for videos on the topic of AI uploaded between 2017 and 2023
start_date = datetime(2017, 1, 1)
end_date = datetime(2017, 12, 31)
videos_df = search_videos("AI", published_after=start_date, published_before=end_date)


print(videos_df)