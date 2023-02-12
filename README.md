# Interaction Graph-Based Community Recommendation on Reddit Users and Subreddit

## Project Summary:

This capstone project focuses on graph-based recommender systems for the social media platform Reddit. Users can choose to comment, subscribe, or otherwise interact in different online communities within Reddit called subreddits. Utilizing the graph database and analytics software TigerGraph, we create a recommendation model that recommends subreddits to users based on a variety of different interaction-related features. 

## Project Organization:
Due to the nature of our project, there will be no test target. Currently, the hierarchy of our project goes as follows:

```
Project/
├─ notebooks/
│  ├─ eda.ipynb
│  ├─ model_testing.ipynb
│  ├─ network_stats.ipynb
├─ src/
│  ├─ dataset/
│  │  ├─ create_dataset.py
│  │  ├─ generate_dataset.py
│  ├─ models/
│  │  ├─ gsql/
│  │  │  ├─ tg_means.gsql
│  │  │  ├─ tg_means_sub.gsql
│  │  ├─ cosine_knn.py
│  │  ├─ jaccard.py
│  │  ├─ model.py
│  │  ├─ popular_recommender.py
│  ├─ util/
│  │  ├─ logger_util.py
│  │  ├─ tigergraph_util.py
├─ .gitignore
├─ README.md
├─ run.py
```

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

### Graph Schema Using GraphStudio

#### What is a graph schema?

  A graph schema is a kind of blueprint that defines the types of nodes and edges in the graph data structure, as well as the relationships and constraints between them. TigerGraph has a graphical user interface called GraphStudio that can be used to set up the initial schema and data mapping/loading. [Here](https://docs.tigergraph.com/gsql-ref/current/ddl-and-loading/defining-a-graph-schema#:~:text=A%20graph%20schema%20is%20a,(properties)%20associated%20with%20it) is a useful link that goes more in depth in terms of defining and loading a graph using TigerGraph. [This](https://www.youtube.com/watch?v=Q0JUkiU0lbs) is another short video demonstration showing how to create a schema in GraphStudio.
  
#### Nodes

This graph is heterogeneous, meaning that there are multiple classes of nodes/vertices involved: class “user”, class “subreddit”, and class “comment”. Each class of vertex has their own attributes associated with them, some of which are already existing from the original features of the data and some that are created during feature engineering. The attributes of our vertices can be found below: 

user
![user](https://user-images.githubusercontent.com/71921141/218294775-498e8fc5-dc21-4321-8367-37777dec8a2d.png)

subreddits
![subreddits](https://user-images.githubusercontent.com/71921141/218294707-0d1667b3-fda0-4916-be4f-6b784192e7da.png)

comments
![comments](https://user-images.githubusercontent.com/71921141/218294706-0601545d-85f5-4bdb-a9ac-568c8b8468cb.png)


#### Edges

Similarly, edges can also have attributes associated with them but are instead used to describe relationships between vertices. Our graph has four types of edges: “interacted_with”, “posted”, “replied_to”, and “belongs_to”. Here are some images of our edge types/attributes:

posted
![posted](https://user-images.githubusercontent.com/71921141/218294718-e57f87ea-da7e-496e-8a13-851dd09d6728.png)

interacted_with
![interacted_with](https://user-images.githubusercontent.com/71921141/218294719-b8dfb1cc-f2c6-4c98-be5d-b44033cbca1f.png)

replied_to
![replied_to](https://user-images.githubusercontent.com/71921141/218294720-ffba3ae0-3308-42d7-a028-4f72d4b83c38.png)

belongs_to
![belongs_to](https://user-images.githubusercontent.com/71921141/218294721-1af356af-53c8-4632-84f5-9a922128860b.png)


### Data

To download the data, simply run `py run.py data`. This will download the necessary files and parse them into the graph representation that this project uses. Please note: this data is reasonably large.


<!-- #### TigerGraph Specifics:
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
For info on `my-cert.txt`, please refer to [this guide](https://dev.tigergraph.com/forum/t/tigergraph-python-connection-issue/2776) -->


<!-- ### Data Generation: -->
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
