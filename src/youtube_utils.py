from urllib.parse import urlparse, parse_qs

def extract_video_id(url:str) -> str:
    url = url.strip()   

    parsed_url = urlparse(url)
    # https://www.youtube.com/watch?v=Gfr50f6ZBvo
    # │       │                │     │
    # │       │                │     └── query(v->paramter name)
    # │       │                └──────── path
    # │       └───────────────────────── hostname
    # └────────────────────────────────── scheme


    # Normal YouTube URL
    # https://www.youtube.com/watch?v=Gfr50f6ZBvo
    if parsed_url.hostname in [ "www.youtube.com","youtube.com","m.youtube.com"]:
        query_params = parse_qs(parsed_url.query) 
        # qs -> query string
        # returns a dict {'v': ['BATmIwHORpU']}

        video_id = query_params["v"]    # contains a list

        if video_id:
            return video_id[0]   # taking the 1st element

    # Short YouTube URL
    # https://youtu.be/Gfr50f6ZBvo
    if parsed_url.hostname == "youtu.be":
        video_id = parsed_url.path.strip("/")
        return video_id


    # Invalid URL
    raise ValueError(
        "Invalid YouTube URL."
        "Please enter a valid Youtube video URL"
    )




# YouTube URL
#      │
#      ▼
#    urlparse()
#      │
#      ├── youtube.com
#      │       │
#      │       ▼
#      │   query → ?v=VIDEO_ID
#      │
#      └── youtu.be
#              │
#              ▼
#           path → /VIDEO_ID