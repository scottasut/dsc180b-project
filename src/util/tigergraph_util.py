import pyTigerGraph as tg
import json

def connection(config_path: str):
    """Establishes and returns a TigerGraph connection according to credentials
    in a TigerGraph configuration file.

    Args:
        config_path (str): The path of the TigerGraph configuration file

    Returns:
        pyTigerGraph.TigerGraphConnection: A TigerGraph connection
    """
    with open(config_path) as config:
        args = json.load(config)
    
    conn = tg.TigerGraphConnection(
        host=args["host"],
        graphname=args["graphname"],
        username=args["username"],
        password=args["password"],
        gsqlSecret=args["gsqlSecret"]
    )
    conn.getToken(args['gsqlSecret'])
    return conn
    
def load_graph(conn: tg.TigerGraphConnection, batch_size: int):
    """Loads a TigerGraph graph in batches for model training. Assumes schema used for this
    project.

    Args:
        conn (pyTigerGraph.TigerGraphConnection): A TigerGraph connection
        batch_size (int): size of batches to load data in
    """
    return conn.gds.graphLoader(
        batch_size=batch_size,
        v_in_feats={
            'user': ['fastrp_embedding']
        },
        e_in_feats={
            'interacted_with': ['times'],
            'commented_in': ['times']
        },
        v_extra_feats={
            'user': ['is_train', 'is_test'],
            'subreddit': ['is_train', 'is_test']
        },
        e_extra_feats={
            "interacted_with": ['is_train', 'is_test'],
            'commented_in': ['is_train', 'is_test']
        },
        output_format = 'PyG'
    )
    