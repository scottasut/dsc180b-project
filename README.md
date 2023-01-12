# Interaction Graph-Based Community Recommendation on Reddit Users and Subreddit

### Setup:


### Data Generation:
In order to run this project, you will need two things for the data generation process: A Reddit account and a Kaggle account. If do not have one of these or both you can create them [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) and [here](https://www.reddit.com/register/) respectively. This is crucial in order to use the APIs provided by each of these platforms to procure the data needed for this project.

Once you have done this, follow these steps:

*Reddit* - Go to the [Reddit Authorized Applications Page](https://www.reddit.com/prefs/apps) while signed in. Select 'create another app...', select 'script' and fill in the 'name', 'description', and 'redirect uri' fields. Click 'create app'. Finally, keep track of the personal use script and secret key generated for you--these are needed for API calls! In the file 'configs/config.json' fill in the following fields:
- reddit_username: your login username for Reddit
- reddit_password: your password for reddit
- reddit_client_id: the personal use script code
- reddit_secret_key: the secret key
- reddit_user_agent: ios:DSC180B:v1.0.0 (by /u/<YOUR_REDDIT_USERNAME>)

*Kaggle* - [This](https://github.com/Kaggle/kaggle-api#api-credentials) guide is helpful. Once you have your `kaggle.json` file, place it at `~/.kaggle`. If this directory does not exist (it likely will not), please create it first.

In order to help guide the users and subreddits we wanted to include, we used the following dataset: https://www.kaggle.com/datasets/timschaum/subreddit-recommender?select=subreddit_info.csv

## Resources:
- [Reddit API](https://www.reddit.com/dev/api/)
- [Reddit Network Explorer](https://github.com/memgraph/reddit-network-explorer)
