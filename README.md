<h1 align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/1/18/UCSD_Seal.png", width=150, height=150>
<img src="https://avatars.githubusercontent.com/u/71526309?s=280&v=4", width=150, height=150>
<img src="https://logodownload.org/wp-content/uploads/2018/02/reddit-logo-16.png", width=150, height=150>

Interaction Graph-Based Community Recommendation on Reddit
</h1>

#### Group Members

- Scott Sutherland (sasuther@ucsd.edu)
- Ryan Don (rdon@ucsd.edu)
- Felicia Chan (f4chan@ucsd.edu)
- Pravar Bhandari (psbhanda@ucsd.edu)

## Overview:
<br/>
<div align="center">
<img src="https://user-images.githubusercontent.com/55766484/224842781-a9657aef-54d5-4305-8a09-fce7112693a1.png"  width="600" height="300">
</div><br/>

This capstone project focuses on graph-based recommender systems for the social media platform Reddit. Users can choose to comment, subscribe, or otherwise interact in different online communities within Reddit called subreddits. Utilizing the graph database and analytics software TigerGraph, we create a recommendation model that recommends subreddits to users based on a variety of different interaction-related features.

The source code for the project is broken up as follows:
- `src/dataset`: files which handle data downloading and parsing it into a heterogeneous graph representation.
- `src/features`: files which handles the non-graph feature generation process for our graph data (users/subreddits).
- `src/models`: our baseline and final models which actually make recommendations for users as well as an evaluation handler class. `src/models/baselines.py` contains the non-graph baseline models while `src/models/models.py` contains the graph-based final models. `src/models/evaluator.py` handles evaluation of recommendations via precision@k calculation given a testing interaction set.

The website associated to this project can be found [here](https://scottasut.github.io/dsc180b-project/).

Due to this project's reliance on TigerGraph's tools, our models cannot be run via a test target without access to a cluster. For a quick demo that our code which makes recommendations for a user, please refer to [this video](https://youtu.be/mfJwbF27YR0). Additionally, you may refer to `notebooks/model_testing.ipynb` to see the evaluation of the models.

## Project Structure:
```
dsc180b-project/
├─ docs/
│  ├─ css/
│  ├─ images/
│  ├─ _config.yml
│  ├─ index.html
├─ notebooks/
│  ├─ eda.ipynb
│  ├─ model_testing.ipynb
│  ├─ network_stats.ipynb
│  ├─ test.ipynb
├─ src/
│  ├─ dataset/
│  │  ├─ create_dataset.py
│  │  ├─ generate_dataset.py
│  │  ├─ make_dataset.py
│  ├─ features/
│  │  ├─ build_features.py
│  ├─ models/
│  │  ├─ baselines.py
│  │  ├─ evaluator.py
│  │  ├─ model.py
│  │  ├─ models.py
│  ├─ util/
│  │  ├─ logger_util.py
│  │  ├─ tigergraph_util.py
├─ .gitignore
├─ Dockerfile
├─ README.md
├─ poster.pdf
├─ report.pdf
├─ run.py
├─ submission.json
```

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

Within GraphStudio, you can follow [these](https://www.youtube.com/watch?v=7sg6Cw7BuWw) steps

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

<!-- #### Nodes

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
![belongs_to](https://user-images.githubusercontent.com/71921141/218294721-1af356af-53c8-4632-84f5-9a922128860b.png) -->

#### Important Usage Notes:
- Your TigerGraph cluster must be on when calling any of the functions here which use `pyTigerGraph` otherwise a connection will not be able to be established. If you are experiencing connection errors, ensure that the cluster you are using is indeed turned on.
- The `data` and `feature` target processes can be configured in a couple of ways via a mandatory file `configs/setup.json` which contains the following where `year`, `month`, `test_year`, `test_month` specify the years and months which training and testing data should be pulled from Reddit respectively and `keywords` specifies the number of keywords we save from a user's comment history (and by extension the size of their keyword embeddings). *Note that more recent data within Reddit is larger and will increase the computational needs for almost every aspect of the project. To see where the data is pulled from and see the file sizes, please refer [here](https://files.pushshift.io/reddit/comments/)*.
```
{
    "year": "2010",
    "month": "12",
    "test_year": "2011",
    "test_month": "03",
    "keywords": 25
}
```

## Resources:
- [Course Site](https://dsc-capstone.github.io/)
- [Project Specifications](https://dsc-capstone.github.io/assignments/projects/q2/)
- [TigerGraph](https://www.tigergraph.com/)
- [TigerGraph Cloud](https://tgcloud.io/)
- [Reddit Comment Datasets](https://files.pushshift.io/reddit/comments/)
- [TigerGraph Community ML Algos](https://docs.tigergraph.com/graph-ml/current/community-algorithms/)
