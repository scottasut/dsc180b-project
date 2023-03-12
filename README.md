<h1 align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/1/18/UCSD_Seal.png", width=150, height=150>
<img src="https://avatars.githubusercontent.com/u/71526309?s=280&v=4", width=150, height=150>
<img src="https://logodownload.org/wp-content/uploads/2018/02/reddit-logo-16.png", width=150, height=150>

Interaction Graph-Based Community Recommendation on Reddit
</h1>

**Authors**

- Scott Sutherland (sasuther@ucsd.edu)
- Ryan Don (rdon@ucsd.edu)
- Felicia Chan (f4chan@ucsd.edu)
- Pravar Bhandari (psbhanda@ucsd.edu)

## Overview:

This capstone project focuses on graph-based recommender systems for the social media platform Reddit. Users can choose to comment, subscribe, or otherwise interact in different online communities within Reddit called subreddits. Utilizing the graph database and analytics software TigerGraph, we create a recommendation model that recommends subreddits to users based on a variety of different interaction-related features.

For the code checkpoint, we do not have a test target due to reliance on TigerGraph. For a quick demo that our code up to this point is working, please refer to [this video.](https://youtu.be/mfJwbF27YR0)

## Prerequisites:

Beyond the packages outlines in `requirements.txt`, there are a few tools needed for this project. Namely:
- [wget](https://www.gnu.org/software/wget/): In order to download the data, you will need wget installed on your system.
  - If you do not meet this requirement, [here](https://www.jcchouinard.com/wget/) is a useful guide on how you can get it.
- [TigerGraph](https://www.tigergraph.com/): To get this setup for this project there are quite a few steps. Let's walk through them:

### Working with TigerGraph:<a name="workingwithtigergraph"></a>

#### Setting up a TGCloud account
In order to leverage graph-based machine learning techniques and TigerGraph's suite of tools in particular, we need to set up a TGCloud instance to work from:

1. Got to https://tgcloud.io/.
2. Select 'sign up'.
3. Fill in the requested information on the sign up page. The organization name can be anything you like, but you will need it to log in.
4. Log in using the information you just provided.
5. Select 'Clusters' on the left hand side menu bar, and then select 'Create Cluster' in the upper right of the interface.
6. From here, you can choose how to configure your cluster. We were able to achieve all the goals of this project using a free cluster with the following specifications: Version: `3.8`, Instance type: `4 vCPU, 7.5GB Memory`, Storage: `50GB`, Number of nodes: `1 Node, Partition Factor 1, Replication Factor 1`.

#### Defining a Graph and Graph Schema
Next, within the GraphStudio tool, we need to create a graph schema to hold our data:

1. Nagivate to 'Tools' > 'GraphStudio' and select the cluster you created.
2. Follow [these](https://youtu.be/Z48cjYuJXX4) steps to create the required schema:


#### What is a graph schema?

A graph schema is a kind of blueprint that defines the types of nodes and edges in the graph data structure, as well as the relationships and constraints between them. TigerGraph has a graphical user interface called GraphStudio that can be used to set up the initial schema and data mapping/loading. [Here](https://docs.tigergraph.com/gsql-ref/current/ddl-and-loading/defining-a-graph-schema#:~:text=A%20graph%20schema%20is%20a,(properties)%20associated%20with%20it) is a useful link that goes more in depth in terms of defining and loading a graph using TigerGraph. [This](https://www.youtube.com/watch?v=Q0JUkiU0lbs) is another short video demonstration showing how to create a schema in GraphStudio.

#### Loading our Data:
*This step requires that the data has been downloaded and processed, please refer to the [Usage](#usage) section*

Within GraphStudio, you can follow these steps:
TODO: Add video of mapping data

Once you have TigerGraph up and running, you need to be able to authenticate yourself when using it. In this project, you can do so by creating `configs/tigergraph_config.json` in this directory which contains the following: 
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
While we worked in TigerGraph, we needed to have a file `my-cert.txt` located in our local machine's root directory `~`. Please refer to [this](https://dev.tigergraph.com/forum/t/tigergraph-python-connection-issue/2776) thread for information on how to get that file.

## Usage:<a name="usage"></a>
In order to run the different components of the project, you will interact with the `run.py` file. There are three main 'targets' or arguments you can pass to the script when running it to work with the project: `data`, `features`, `...`

- `data`: downloads the raw data and parses it into a heterogeneous graph format
- `features`: generates necessary features for final model from raw data. Depends on the `data` target.
- `...`: runs the final model. Depends on `feature` target and [data loaded into your TigerGraph cluster](#workingwithtigergraph).

Targets can be called as follows `python run.py data features ...`.

#### Nodes

This graph is heterogeneous, meaning that there are multiple classes of nodes/vertices involved: class “user”, class “subreddit”, and class “comment”. Each class of vertex has their own attributes associated with them, some of which are already existing from the original features of the data and some that are created during feature engineering. The attributes of our vertices can be found below: 

<!-- user
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
![belongs_to](https://user-images.githubusercontent.com/71921141/218294721-1af356af-53c8-4632-84f5-9a922128860b.png) -->

## Some important notes:
- Your TigerGraph cluster must be on when calling any of the functions here which use `pyTigerGraph` otherwise a connection will not be able to be established. If you are experiencing connection errors, ensure that the cluster you are using is indeed turned on.

## Resources:
- [Reddit API](https://www.reddit.com/dev/api/)
- [Reddit Network Explorer](https://github.com/memgraph/reddit-network-explorer)
- [Reddit Comment Datasets](https://files.pushshift.io/reddit/comments/)
- [TigerGraph Community ML Algos](https://docs.tigergraph.com/graph-ml/current/community-algorithms/)
