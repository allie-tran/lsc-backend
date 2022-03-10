from gensim.models.phrases import Phraser
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
from nltk import pos_tag
from collections import defaultdict
from ..nlp_utils.common import *
from ..nlp_utils.pos_tag import *
from ..nlp_utils.time import *
from ..nlp_utils.synonym import *
import numpy as np
init_tagger = Tagger(locations)
time_tagger = TimeTagger()
e_tag = ElementTagger()

bigram_phraser = Phraser.load("/home/tlduyen/LSC2020/files/bigram_phraser.pkl")

def process_for_ocr(text):
    final_text = defaultdict(lambda : defaultdict(float))
    for word in text:
        final_text[word][word] = 1
        for i in range(0, len(word)-1):
            if len(word[:i+1]) > 1:
                final_text[word][word[:i+1]] += (i+1) / len(word)
            if len(word[i+1:]) > 1:
                final_text[word][word[i+1:]] += 1 - (i+1)/len(word)
    return final_text

def search(wordset, text):
    results = []
    for keyword in wordset:
        if keyword:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                results.append(keyword)
    return results

def search_location(text):
    results = []
    gps_results = []
    for location in locations:
        for i, extra in enumerate(locations[location]):
            # Full match
            if i == 0 and re.search(r'\b' + re.escape(extra) + r'\b', text, re.IGNORECASE):
                return [location], [(location, 1.0)], True
            if re.search(r'\b' + re.escape(extra) + r'\b', text, re.IGNORECASE):
                if extra not in results:
                    results.append(location)
                gps_results.append((location, len(extra.split())/len(location.split())))
                break
    return results, gps_results, False

gps_location_sets = {location: set([pl for pl in location.lower().replace(',', ' ').split() if pl not in stop_words]) for location, gps in map_visualisation}
gps_not_lower = {}
for loc in locations:
    for origin_doc, (lat, lon) in map_visualisation:
        if loc == origin_doc.lower():
            gps_not_lower[loc] = origin_doc

class Query:
    def __init__(self, text, shared_filters=None):
        self.negative = ""
        if "NOT" in text:
            text, self.negative = text.split("NOT")
            self.negative = self.negative.strip(". \n").lower()
            self.negative = [word for word in self.negative.split() if word in all_keywords]
        text = text.strip(". \n").lower()
        self.time_filters = None
        self.date_filters = None
        self.ocr_queries = []
        self.location_queries = []
        self.query_visualisation = defaultdict(list)
        self.location_filters = []
        self.country_to_visualise = []
        self.extract_info(text, shared_filters)

    def extract_info(self, text, shared_filters=None):
        def search_words(wordset):
            return search(wordset, text)
        self.original_text = text

        quoted_text = " ".join(re.findall(r'\"(.+?)\"', text))
        text = text.replace(f'"{quoted_text}"', "") #TODO!

        self.ocr = process_for_ocr(quoted_text.split())
        self.regions = search_words(regions)

        for reg in self.regions:
            self.query_visualisation["REGION"].append(reg)
            for country in countries:
                if reg == country.lower():
                    self.country_to_visualise.append(country)

        self.locations, self.gps_results, full_match = search_location(text)
        processed = set([w for word in self.regions +
                         self.locations for w in word.split()])
        self.place_to_visualise = [gps_not_lower[location] for location, score in self.gps_results]

        if not full_match:
            # self.locations.extend(search_words(
                # [w for w in ["hotel", "restaurant", "airport", "station", "cafe", "bar", "church"] if w not in self.locations]))

            for loc in self.locations[len(self.gps_results):]:
                for place, _ in map_visualisation:
                    if loc in place.lower().split():
                        self.place_to_visualise.append(place)
        if full_match:
            for loc in self.locations:
                self.query_visualisation["LOCATION"].append(loc)
        else:
            for loc in self.locations:
                self.query_visualisation["POSSIBLE LOCATION"].append(loc)

        self.weekdays = []
        self.dates = None
        self.start = (0, 0)
        self.end = (24, 0)

        tags = time_tagger.tag(text)
        print(tags)
        for i, (word, tag) in enumerate(tags):
            if tag in ["WEEKDAY", "TIMERANGE", "TIMEPREP", "DATE", "TIME", "TIMEOFDAY"]:
                processed.update(word.split())
                self.query_visualisation["TIME" if "TIME" in tag else tag].append(word)
            if tag == "WEEKDAY":
                self.weekdays.append(word)
            elif tag == "TIMERANGE":
                s, e = word.split("-")
                self.start = adjust_start_end(
                    "start", self.start, *am_pm_to_num(s))
                self.end = adjust_start_end("end", self.end, *am_pm_to_num(e))
            elif tag == "TIME":
                timeprep = ""
                if i > 1 and tags[i-1][1] == 'TIMEPREP':
                    timeprep = tags[i-1][0]
                if timeprep in ["before", "earlier than", "sooner than"]:
                    self.end = adjust_start_end(
                        "end", self.end, *am_pm_to_num(word))
                elif timeprep in ["after", "later than"]:
                    self.start = adjust_start_end(
                        "start", self.start, *am_pm_to_num(word))
                else:
                    h, m = am_pm_to_num(word)
                    self.start = adjust_start_end(
                        "start", self.start, h - 1, m)
                    self.end = adjust_start_end("end", self.end, h + 1, m)
            elif tag == "DATE":
                self.dates = get_day_month(word)
            elif tag == "TIMEOFDAY":
                timeprep = ""
                if i > 1 and tags[i-1][1] == 'TIMEPREP':
                    timeprep = tags[i-1][0]
                if "early" in timeprep:
                    if "early; " + word in timeofday:
                        word = "early; " + word
                elif "late" in timeprep:
                    if "late; " + word in timeofday:
                        word = "late; " + word
                if word in timeofday:
                    s, e = timeofday[word].split("-")
                    self.start = adjust_start_end(
                        "start", self.start, *am_pm_to_num(s))
                    self.end = adjust_start_end(
                        "end", self.end, *am_pm_to_num(e))
                else:
                    print(
                        word, f"is not a registered time of day ({timeofday})")
        if shared_filters:
            if not self.weekdays:
                self.weekdays.extend(shared_filters.weekdays)
            if self.dates is None:
                self.dates = shared_filters.dates
        unprocessed = [(word, tag) for (word, tag) in tags if word not in processed]

        last_non_prep = 0
        for i in range(1, len(unprocessed)):
            if unprocessed[-i][1] != "IN":
                last_non_prep = i
                break
        if last_non_prep > 1:
            self.clip_text = " ".join([word for word, tag in unprocessed[:-(last_non_prep - 1)]])
        else:
            self.clip_text = " ".join(
                [word for word, tag in unprocessed])
        print("CLIP:", self.clip_text)
        # self.query_visualisation[self.clip_text] = "CLIP"

    def get_info(self):
        return {"query_visualisation": [(hint, ", ".join(value)) for hint, value in self.query_visualisation.items()],
                "country_to_visualise": self.country_to_visualise,
                "place_to_visualise": self.place_to_visualise}

    def time_to_filters(self):
        if not self.time_filters:
            # Time
            self.time_filters = {
                                    "range":
                                    {
                                        "hour":
                                        {
                                            "gte": self.start[0],
                                            "lte": self.end[0]
                                        }
                                    }
                                }

            # Date
            self.date_filters = []
            if self.dates:
                y, m, d = self.dates
                if y:
                    self.date_filters.append({"term": {"year": str(y)}})
                if m:
                    self.date_filters.append(
                        {"term": {"month": str(m).rjust(2, "0")}})
                if d:
                    self.date_filters.append(
                        {"term": {"date": str(d).rjust(2, "0")}})

        return self.time_filters, self.date_filters

    def make_ocr_query(self):
        if not self.ocr_queries:
            self.ocr_queries = []
            for ocr_word in self.ocr:
                dis_max = []
                for ocr_word, score in self.ocr[ocr_word].items():
                    dis_max.append(
                        {"rank_feature": {"field": f"ocr_score.{ocr_word}", "boost": 200 * score, "linear": {}}})
                self.ocr_queries.append({"dis_max": {
                    "queries": dis_max,
                    "tie_breaker": 0.0}})
        return self.ocr_queries
        #TODO: multiple word in OCR

    def make_location_query(self):
        if not self.location_queries:
            # Matched GPS
            for loc, score in self.gps_results:
                place = gps_not_lower[loc]
                print(place)
                dist = "0.5km"
                pivot = "5m"
                if "airport" in loc or "home" in loc:
                    dist = "2km"
                    pivot = "200m"
                elif "dcu" in loc:
                    dist = "1km"
                    pivot = "100m"
                
                for place_iter, (lat, lon) in map_visualisation:
                    if place == place_iter:
                        self.location_queries.append({
                                "distance_feature": {
                                    "field": "gps",
                                    "pivot": pivot,
                                    "origin": [lon, lat],
                                    "boost": score * 50
                                }
                            })

                        self.location_filters.append({
                            "geo_distance": {
                                "distance": dist,
                                "gps": [lon, lat]
                            }
                        })
                        break

            # General:
            if len(self.gps_results) < len(self.locations):
                for loc in self.locations[len(self.gps_results):]:
                    loc_set = set(loc.split())
                    for place, (lat, lon) in map_visualisation:
                        set_place = gps_location_sets[place]
                        if loc_set.issubset(set_place):
                            pivot = "5m"
                            if "airport" in set_place:
                                pivot = "200m"
                                self.location_filters.append({
                                    "geo_distance": {
                                        "distance": "2km",
                                        "gps": [lon, lat]
                                    }
                                })
                            elif "dcu" in set_place:
                                pivot = "100m"
                            self.location_queries.append({
                                "distance_feature": {
                                    "field": "gps",
                                    "pivot": pivot,
                                    "origin": [lon, lat],
                                    "boost": len(loc_set) / len(set_place) * 50
                                }
                            })
        if self.location_queries:
            return {"dis_max": {"queries": self.location_queries, "tie_breaker": 0.0}}
        else:
            return None
