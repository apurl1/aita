import praw
from psaw import PushshiftAPI
import pandas as pd
import time

# obtain reddit instance
reddit = praw.Reddit(client_id='pS9Tq9o2dYRgVg', client_secret='XHAJBe9LSub-yD8QOIjoz5n882o', user_agent='my user agent')
api = PushshiftAPI(reddit)

# obtain generator list of last 25000 submissions
gen = api.search_submissions(subreddit='AmItheAsshole', limit=25000)

# create dictionary for data
data = {'Title': [], 'Text': [], 'Score': [], 'Upvote Ratio': [], 
        'Num Awards': [], 'Num Comments': [], 'Num Crossposts': [], 
        'Label': [], 'Author Flair': []}

# check that posts are not NSFW and are marked with a flair
def satisfies_criteria(sub):
    if sub.link_flair_text is None:
        return False
    if sub.link_flair_text is "":
        return False
    if sub.over_18:
        return False
    return True

# parse and save data from posts obtained
i = 0
for sub in gen:
    print(i)
    i += 1
    if satisfies_criteria(sub):
        data['Title'].append(sub.title)
        data['Text'].append(sub.selftext)
        data['Score'].append(sub.score)
        data['Upvote Ratio'].append(sub.upvote_ratio)
        data['Num Awards'].append(sub.total_awards_received)
        data['Num Comments'].append(sub.num_comments)
        data['Num Crossposts'].append(sub.num_crossposts)
        data['Label'].append(sub.link_flair_text)
        data['Author Flair'].append(sub.author_flair_text)
        # data['Author Karma'].append(reddit.redditor(sub.author).link_karma)

# save data to csv
df = pd.DataFrame(data)
df.to_csv('data.csv')
