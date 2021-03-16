# Created on Mar 3, 2021 Tue. by Guanglin Du
# See https://neo4j.com/docs/api/python-driver/current/api.html
# See also https://towardsdatascience.com/neo4j-cypher-python-7a919a372be7

# pip install neo4j

from neo4j import GraphDatabase
from neo4j import __version__ as neo4j_version

uri = "neo4j://192.168.1.2:7687"
passwd = "your_password" # please modify according to your needs


def before_test():
    print("neo4j driver version: " + neo4j_version) # 4.2.1
    driver = GraphDatabase.driver(uri, auth=("neo4j", passwd), max_connection_lifetime=1000)
    driver.close()


class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri,
                auth=(self.__user, self.__pwd),  max_connection_lifetime=1000)
        except Exception as e:
            print("Failed to create the driver:", e)


    def close(self):
        if self.__driver is not None:
            self.__driver.close()


    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response


if __name__ == "__main__":
    # before_test()
    n4jc = Neo4jConnection(uri, "neo4j", passwd)

    q1 = "MATCH (n) RETURN count(n)"
    print(n4jc.query(q1, "neo4j"))

    q1 = "MATCH (p:Person) WHERE p.name CONTAINS '瑞' RETURN p"
    print(n4jc.query(q1, "neo4j"))
