from django.http import JsonResponse
from datetime import datetime
from pymongo import MongoClient
import random
from django.views.decorators.csrf import csrf_exempt


client = MongoClient()
db = client.lsc.session
# db.drop()
# db = client.lsc.session
db2 = client.lsc.query
db2.drop()
db2 = client.lsc.query
SECONDS_PER_CLUE = 60
MAX_POINT = 100
MAX_POINT_TASK_END = 50
PENALTY_PER_WRONG = 10
TEST_QUERIES = [73, 66, 62]
EXP_QUERIES = [72, 64, 57, 63, 74]

class Query():
    def __init__(self, idx, text=[], results=[]):
        self.idx = idx
        existed = db.find_one({"idx": idx})
        if existed:
            self.restore_from_dict(existed)
        else:
            self.id = None
            self.text = text # ["Default Query 1", "Default Query 2", "Default Query 3"] 
            self.results = results # ["image1", "image2", "image3"]
            self.current = 0
            self.write_to_db()
    
    def restore_from_dict(self, dictdata):
        self.text = dictdata["text"]
        self.results = dictdata["results"]
        self.current = dictdata["current"]
        self.idx = dictdata["idx"]
        self.id = dictdata["_id"]

    def to_dict(self):
        return {"text": self.text, 
                "results": self.results, 
                "current": self.current}
    
    def get_current_text(self):
        return self.text[self.current]

    def restart(self):
        self.current = 0
        self.write_to_db()

    def next_clue(self):
        self.current += 1
        self.write_to_db()
        if self.current >= len(self.text):
            return False
        return True
    
    def finish_clue(self):
        self.current = len(self.text) - 1
        self.write_to_db()

    def eval(self, imageid):
        return imageid in self.results

    def write_to_db(self):
        if self.id:
            db2.update_one({'_id' : self.id}, {'$set': self.to_dict()})
        else:
            self.id = db2.insert_one(self.to_dict()).inserted_id

def get_all_queries(filename):
    query_id = None
    text = []
    results = []
    queries = {}
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                if query_id:
                    queries[query_id] = Query(query_id, text, results)
                query_id = None
                text = []
                results = []
                continue
            if len(line) == 2:
                query_id = int(line)
            else:
                if len(text) < 6:
                    text.append(line.strip().replace('. ', '.\n'))
                else:
                    results.append(line.strip())
    if query_id:
        queries[query_id] = Query(query_id, text, results)
    return queries

ALL_QUERIES = get_all_queries('backend/all_queries.txt')
print(ALL_QUERIES.keys())

# declare a new model with a name "GeeksModel" 
class LSCSession():
    def __init__(self, name):
        self.name = name
        existed = db.find_one({"name": name})
        if existed:
            self.restore_from_dict(existed)
        else:
            self.time = 0
            self.submissions = [[] for i in range(5)]
            self.id = None
            if "test" in name.lower():
                self.query_ids = TEST_QUERIES
            else:
                self.query_ids = EXP_QUERIES
            self.query_id = 0
            self.scores = [0 for i in range(5)]
            self.write_to_db()

    def reset(self):
        self.time = 0
        self.submissions = [[] for i in range(5)]
        self.id = None
        if "test" in name.lower():
            self.query_ids = TEST_QUERIES
        else:
            self.query_ids = EXP_QUERIES
        self.query_id = 0
        self.scores = [0 for i in range(5)]
        self.write_to_db()

    def add_submission(self, imageid):
        current_query = self.get_current_query()
        correctness = current_query.eval(imageid)
        self.submissions[self.query_id].append((imageid, correctness, (60 - self.time) + current_query.current * 60))
        self.get_score()
        if correctness:
            self.get_current_query().finish_clue()
        self.write_to_db()
        return correctness

    def set_time(self, time):
        self.time = int(time)
        self.write_to_db()

    def get_score(self):
        submissions = self.submissions[self.query_id]
        duration = SECONDS_PER_CLUE * len(self.get_current_query().text)
        correctness = [sub[1] for sub in submissions]
        if True in correctness:
            first_correct = correctness.index(True)
            time_fraction = 1 - min(1.0, submissions[first_correct][2] / duration)
            self.scores[self.query_id] = max(0, MAX_POINT_TASK_END + ((MAX_POINT - MAX_POINT_TASK_END) * time_fraction) - (first_correct * PENALTY_PER_WRONG))
        self.write_to_db()

    def get_current_query(self):
        return ALL_QUERIES[self.query_ids[self.query_id]]
    
    def get_current_score(self):
        return round(self.scores[self.query_id], 2)

    def get_total_score(self):
        return round(sum(self.scores), 2)

    def restore_from_dict(self, dictdata):
        self.name = dictdata["name"]
        self.time = dictdata["time"]
        self.scores = dictdata["scores"]
        self.submissions = dictdata["submissions"]
        self.query_ids = dictdata["query_ids"]
        self.query_id = dictdata["query_id"]
        self.id = dictdata["_id"]

    def to_dict(self):
        return {"name": self.name, 
                "time": self.time, 
                "scores": self.scores, 
                "submissions": self.submissions, 
                "query_ids": self.query_ids,
                "query_id": self.query_id}

    def next_query(self):
        self.time = 0
        self.query_id += 1
        self.write_to_db()
        if self.query_id >= len(self.query_ids):
            return False
        return True

    def write_to_db(self):
        if self.id:
            db.update_one({'_id' : self.id}, {'$set': self.to_dict()})
        else:
            self.id = db.insert_one(self.to_dict()).inserted_id
        
def jsonize(response):
    # JSONize
    response = JsonResponse(response)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response["Access-Control-Allow-Credentials"] = "true"
    response["Access-Control-Allow-Headers"] = "X-Requested-With, Content-Type"
    return response

@csrf_exempt
def new_session(request):
    session_name = request.GET.get("session_name")
    session = LSCSession(session_name)
    try:
        query = session.get_current_query()
        query.restart()
        return jsonize({"query": query.get_current_text(), "score": session.get_current_score(), "total_score": session.get_total_score()})
    except IndexError:
        return jsonize({"query": "The End.", "score": session.get_current_score(), "total_score": session.get_total_score()})

@csrf_exempt
def end_query_round(request):
    session_name = request.GET.get("session_name")
    session = LSCSession(session_name)
    valid = session.next_query()
    if valid:
        query = session.get_current_query()
        query.restart()
        return jsonize({"query": query.get_current_text()})
    else:
        return jsonize({"query": "The End.", "score": session.get_current_score(), "total_score": session.get_total_score()})

@csrf_exempt
def next_clue(request):
    session_name = request.GET.get("session_name")
    session = LSCSession(session_name)
    query = session.get_current_query()
    print(query.get_current_text())
    valid = query.next_clue()
    if not valid:
        session_valid = session.next_query()
        if session_valid:
            query = session.get_current_query()
            query.restart()
            return jsonize({"query": query.get_current_text()})
        else:
            return jsonize({"query": "The End."})
    return jsonize({"query": query.get_current_text(), "score": session.get_current_score(), "total_score": session.get_total_score()})

@csrf_exempt
def get_query(request):
    session_name = request.GET.get('session_name')
    session = LSCSession(session_name)
    return jsonize({"query": [ALL_QUERIES[idx].text for idx in session.query_ids]})

@csrf_exempt
def submit(request):
    session_name = request.GET.get('session_name')
    session = LSCSession(session_name)
    imageid = request.GET.get('imageid')
    time = request.GET.get('time')
    correctness = session.add_submission(imageid)
    return jsonize({"description": correctness})

@csrf_exempt
def get_score(request):
    session_name = request.GET.get('session_name')
    current_time = request.GET.get('time')
    print(current_time)
    session = LSCSession(session_name)
    session.set_time(current_time)
    return jsonize({"score": session.get_current_score(), "total_score": session.get_total_score()})
