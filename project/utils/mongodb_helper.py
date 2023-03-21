from pymongo import MongoClient
from bson.json_util import dumps
import pprint
import json


class MongoDBHelper:

    """
    Provide basic commands and operations to connect to database, insert, update annotations 
    """

    def __init__(self, mongo_client_ip='localhost', port=27017):
        self._client = MongoClient(mongo_client_ip, port)


    def connect_db(self, db_name):
        self._db = self._client[db_name]
    
    
    def connect_collection(self, collection_name):
        self._collection = self._db[collection_name]


    def query(self, query_condition, field_filter, distinct_by=None, max_return=100000, collection_name=None):
        if collection_name != None:
            self._collection = self._db[collection_name]
        result = self._collection.find(query_condition, field_filter, limit=max_return)
        if distinct_by != None:
            result = result.distinct(distinct_by)
        result = dumps(result[:max_return])
        return result

    def insert_to_collection(self, data, ordered=True, collection_name=None):
        if collection_name != None:
            self._collection = self._db[collection_name]
        self._collection.insert_many(data, ordered=ordered)

    
    def update_one(self, object_id, data, update_field, method='$set', upsert=False, collection_name=None):
        if collection_name != None:
            self._collection = self._db[collection_name]
        self._collection.update_one({ '_id': object_id }, { method : { update_field : data }}, upsert=upsert)

    
    # def update_many(self, object_ids, data, method='$set', upsert=False, collection_name=None):
    #     if collection_name != None:
    #         self._collection = self._db[collection_name]
    #     self._collection.update_many({'$in' : {'_id' : object_ids }}, { method : { data } }, { 'upsert': upsert })