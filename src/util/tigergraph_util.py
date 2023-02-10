import pyTigerGraph as tg
import json

def connect():
    with open("../../config.json", "r") as config:
        args = json.load(config)
    
    conn = tg.TigerGraphConnection(
        host=args["host"],
        graphname=args["graphname"],
        username=args["username"],
        password=args["password"],
        gsqlSecret=args["gsqlSecret"]
    )
    conn.getToken(args["gsqlSecret"])
    if not connected(conn):
        raise ConnectionError("Failed to conenct to GSQL")
    return conn

def connected(conn):
    if not conn.echo() == "Hello GSQL":
        return False
    return True
