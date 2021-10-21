import json
from pymongo import MongoClient


anno_file = '/home/tlduyen/LQA/anno_qa.json'
with open(anno_file) as f:
    data = json.load(f)["_default"]

client = MongoClient()
db2 = client.pymongo_test.posts
# for qa in data:
#     doc = db.posts.find_one({"scene": data[qa]["scene"], "i": data[qa]["i"]})
#     if doc is None:
#         print("New!")
#         print(data[qa].keys())
#         db.posts.insert_one(data[qa])
#     else:
#         db.posts.find_one_and_update({"_id" : doc["_id"]}, {"$set":data[qa]}, upsert=True)
#         doc = db.posts.find_one({"scene": data[qa]["scene"], "i": data[qa]["i"]})

# posts = db.posts
# post_data = {
#     'title': 'Python and MongoDB',
#     'content': 'PyMongo is fun, you guys',
#     'author': 'Scott'
# }
# result = posts.insert_one(post_data)
# print('One post: {0}'.format(result.inserted_id))

# bills_post = posts.find_one({'author': 'Bill'})
# print(bills_post)
print(db2.find({"scene": "2015-03-05_1"}))
scene_qas = sorted(db2.find({"scene": "2015-03-05_1"}), key=lambda qa: qa["i"])
scene_qas = [(qa["i"], qa["qa"]) for qa in scene_qas]
print(scene_qas)
