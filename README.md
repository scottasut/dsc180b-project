# Interaction Graph-Based Community Recommendation on Reddit Users and Subreddit

## Setup:

Before we can get going, there are a few steps we need to take. Making sure we have all the required software and tools is one of them:

### Required Tools:
- TigerGraph: 
  - `config.json`: Once you have TigerGraph up and running, you need to be able to authenticate yourself when using it. In this project, you can do so by creating `config.json` in this directory which has the following format: 
    ```
    {
        "host": "<The Host for your TigerGraph Cluster>",
        "graphname": "<The Name of Your Graph>",
        "username": "<Your TigerGraph Username>",
        "password": "<Your TigerGraph Password>",
        "gsqlSecret": "<Your Secret Key>",
        "certPath": "<The location of your my-cert.txt>"
    }
    ``` 
    While we worked in TigerGraph, we needed to have a file `my-cert.txt` located in our computer's root directory `~`. Please refer to [this](https://dev.tigergraph.com/forum/t/tigergraph-python-connection-issue/2776) for information on how to get that file.
- `wget`: In order to download the data, you will need [wget](https://www.gnu.org/software/wget/) installed on your system. If you do not meet this requirement, [here](https://www.jcchouinard.com/wget/) is a guide on how you can get it.

### Data

To download the data, simply run `py run.py data`. This will download the necessary files and parse them into the graph representation that this project uses. Please note: this data is reasonably large, 


#### TigerGraph Specifics:
For access to TigerGraph, you are going to need to set up a config file with the following information:
```
{
    "host": "<The Host for your TigerGraph Cluster>",
    "graphname": "<The Name of Your Graph>",
    "username": "<Your TigerGraph Username>",
    "password": "<Your TigerGraph Password>",
    "gsqlSecret": "<Your Secret Key>",
    "certPath": "<The location of your my-cert.txt>"
}
```
For info on `my-cert.txt`, please refer to [this guide](https://dev.tigergraph.com/forum/t/tigergraph-python-connection-issue/2776)


### Data Generation:
<!-- In order to run this project, you will need two things for the data generation process: A Reddit account and a Kaggle account. If do not have one of these or both you can create them [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) and [here](https://www.reddit.com/register/) respectively. This is crucial in order to use the APIs provided by each of these platforms to procure the data needed for this project.

Once you have done this, follow these steps:

*Reddit* - Go to the [Reddit Authorized Applications Page](https://www.reddit.com/prefs/apps) while signed in. Select 'create another app...', select 'script' and fill in the 'name', 'description', and 'redirect uri' fields. Click 'create app'. Finally, keep track of the personal use script and secret key generated for you--these are needed for API calls! In the file 'configs/config.json' fill in the following fields:
- reddit_username: your login username for Reddit
- reddit_password: your password for reddit
- reddit_client_id: the personal use script code
- reddit_secret_key: the secret key
- reddit_user_agent: ios:DSC180B:v1.0.0 (by /u/<YOUR_REDDIT_USERNAME>)

*Kaggle* - [This](https://github.com/Kaggle/kaggle-api#api-credentials) guide is helpful. Once you have your `kaggle.json` file, place it at `~/.kaggle`. If this directory does not exist (it likely will not), please create it first.

In order to help guide the users and subreddits we wanted to include, we used the following dataset: https://www.kaggle.com/datasets/timschaum/subreddit-recommender?select=subreddit_info.csv -->

## Resources:
- [Reddit API](https://www.reddit.com/dev/api/)
- [Reddit Network Explorer](https://github.com/memgraph/reddit-network-explorer)
- [Reddit Comment Datasets](https://files.pushshift.io/reddit/comments/)
- [TigerGraph Community ML Algos](https://docs.tigergraph.com/graph-ml/current/community-algorithms/)
