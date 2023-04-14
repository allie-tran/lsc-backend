from collections import defaultdict
from ..nlp_utils.common import *
from ..nlp_utils.time import *
time_tagger = TimeTagger()

def filter_locations(location):
    if location in ["", "the house", "restaurant"]:
        return False
    return True

def search(wordset, text):
    results = []
    text = " " + text + " "
    for keyword in wordset:
        if filter_locations(keyword):
            if keyword:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    results.append(keyword)
    return results

# Partial match only
def search_possible_location(text):
    results = []
    for location in locations:
        if filter_locations(location):
            for i, extra in enumerate(locations[location]):
                if re.search(r'\b' + re.escape(extra) + r'\b', text, re.IGNORECASE):
                    if extra not in results:
                        results.append(location)
    return results

gps_not_lower = {}
for loc in locations:
    for origin_doc, (lat, lon) in map_visualisation:
        if loc == origin_doc.lower():
            gps_not_lower[loc] = origin_doc

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def range_filter(start, end, field, boost=1.0):
    return {
            "range":
            {
                field:
                {
                    "gte": start,
                    "lte": end,
                    "boost": boost
                }
            }
        }

class Query:
    def __init__(self, text, shared_filters=None):
        self.negative = ""
        self.disable_region = False
        if "—disable_region" in text:
            print("Disabling region")
            self.disable_region = True
            text = text.replace("—disable_region", "")
        self.disable_location = False
        if "—disable_location" in text:
            print("Disabling location")
            self.disable_location = True
            text = text.replace("—disable_location", "")
        if "NOT" in text:
            text, self.negative = text.split("NOT")
        text = text.strip(". \n").lower()
        self.time_filters = None
        self.date_filters = None
        self.duration_filters = None
        self.location_queries = []
        self.query_visualisation = defaultdict(list)
        self.location_filters = []
        self.country_to_visualise = []
        self.clip_embedding = None
        self.duration = None
        self.extract_info(text, shared_filters)

    def extract_info(self, text, shared_filters=None):
        def search_words(wordset):
            return search(wordset, text)
        self.original_text = text

        if not self.disable_location:
            self.locations = search_words(locations)
            print("Locations:", self.locations)
            self.place_to_visualise = [gps_not_lower[location] for location in self.locations]
            if self.locations:
                self.query_visualisation["LOCATION"].extend(self.locations)
            # else:
            #     possible_locations = search_possible_location(text)
                # if possible_locations:
                #     self.query_visualisation["POSSIBLE LOCATION(S)"].extend(possible_locations)
        else:
            self.locations = []
            self.place_to_visualise = []

        for loc in self.locations:
            text = rreplace(text, loc, "", 1) #TODO!

        if not self.disable_region:
            self.regions = search_words(regions)
        else:
            self.regions = []

        for reg in self.regions:
            self.query_visualisation["REGION"].append(reg)
            if reg in lowercase_countries:
                country = lowercase_countries[reg]
                self.country_to_visualise.append({"country": country, "geojson": countries[country]})
        for region in self.regions:
            text = rreplace(text, region, "", 1) #TODO!

        # processed = set([w.strip(",.") for word in self.regions +
                        #  self.locations for w in word.split()])
        # if not full_match:
        #     # self.locations.extend(search_words(
        #         # [w for w in ["hotel", "restaurant", "airport", "station", "cafe", "bar", "church"] if w not in self.locations]))
        #     for loc in self.locations[len(self.gps_results):]:
        #         for place, _ in map_visualisation:
        #             if loc in place.lower().split():
        #                 self.place_to_visualise.append(place)

        # if full_match:
        #     for loc in self.locations:
        #         self.query_visualisation["LOCATION"].append(loc)
        # else:
        #     for loc in self.locations:
        #         self.query_visualisation["POSSIBLE LOCATION"].append(loc)

        self.weekdays = []
        self.dates = []
        self.start = (0, 0)
        self.end = (24, 0)

        tags = time_tagger.tag(text)
        processed = set()
        for i, (word, tag) in enumerate(tags):
            if word in processed:
                continue
            if tag in ["WEEKDAY", "TIMERANGE", "TIMEPREP", "DATE", "TIME", "TIMEOFDAY", "PERIOD"]:
                processed.add(word)
                # self.query_visualisation["TIME" if "TIME" in tag else tag].append(word)
            if tag == "WEEKDAY":
                self.weekdays.append(word)
                self.query_visualisation["WEEKDAY"].append(word)
            elif tag == "TIMERANGE":
                s, e = word.split("-")
                self.start = adjust_start_end(
                    "start", self.start, *am_pm_to_num(s))
                self.end = adjust_start_end("end", self.end, *am_pm_to_num(e))
            elif tag == "TIME":
                if word in ["2015", "2016", "2018", "2019", "2020"]:
                    self.dates.append(get_day_month(word))
                else:
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
                self.dates.append(get_day_month(word))
            elif tag == "SEASON":
                for month in seasons[word]:
                    self.dates.append((None, month2num[month], None))
            elif tag == "TIMEOFDAY":
                if word not in ["lunch", "breakfast", "dinner", "sunrise", "sunset"]:
                    processed.add(word)
                # self.query_visualisation["TIME" if "TIME" in tag else tag].append(word)
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
            elif tag == "PERIOD":
                self.duration = parse_period_expression(word)
                self.query_visualisation["DURATION"] = [f"{word}({self.duration}s)"]
                
        print(tags)
        if shared_filters:
            if not self.weekdays:
                self.weekdays.extend(shared_filters.weekdays)
            if self.dates is None:
                self.dates = shared_filters.dates
        unprocessed = [(word, tag) for (word, tag) in tags if word not in processed]

        last_non_prep = 0
        self.clip_text = ""
        for i in range(1, len(unprocessed)  + 1):
            if unprocessed[-i][1] not in ["DT", "IN"] and unprocessed[-i][0] not in stop_words:
                last_non_prep = i
                break
        if last_non_prep > 1:
            self.clip_text = " ".join([word for word, tag in unprocessed[:-(last_non_prep - 1)]])
        else:
            self.clip_text = " ".join(
                [word for word, tag in unprocessed])
            
        # Remove ending words that are stop words from clip_text
        def strip_stopwords(sentence):
            if sentence:
                words = sentence.split()
                for i in reversed(range(len(words))):
                    if words[i].lower() in stop_words:
                        words.pop(i)
                    else:
                        break
                return " ".join(words)
            return ""
        self.clip_text = strip_stopwords(self.clip_text)
        self.clip_text = self.clip_text.strip(", ")
        print("CLIP:", self.clip_text)

    def get_info(self):
        return {"query_visualisation": [(hint, ", ".join(value)) for hint, value in self.query_visualisation.items()],
                "country_to_visualise": self.country_to_visualise,
                "place_to_visualise": self.place_to_visualise}

    def time_to_filters(self):
        if not self.time_filters:
            # Time
            s = self.start[0] * 3600 + self.start[1] * 60
            e = self.end[0] * 3600 + self.end[1] * 60
            print(self.start, self.end, s, e)
            if s <= e:
                # OR (should queries)
                self.time_filters = [range_filter(s, e, "start_seconds_from_midnight"),
                                     range_filter(s, e, "end_seconds_from_midnight")]
            else: # either from midnight to end or from start to midnight
                self.time_filters = [range_filter(0, e, "start_seconds_from_midnight"),
                                     range_filter(0, e, "end_seconds_from_midnight"),
                                     range_filter(s, 24 * 3600, "start_seconds_from_midnight"),
                                     range_filter(s, 24 * 3600, "end_seconds_from_midnight")]
            # combine the date filters using a bool should clause
            self.time_filters = {"bool": {"should": self.time_filters, "minimum_should_match": 1}}
            # Date
            # Preprocess the dates: if there is only one year available, add that year to all dates
            years = set()
            common_year = None
            for date in self.dates:
                years.add(date[0])
            if len(years) == 1:
                common_year = years.pop()
            
            # create a list to store the date filters
            self.date_filters = []
            if self.dates:
                self.query_visualisation["DATE"] = []
                for date in self.dates:
                    y, m, d = date
                    if not y and common_year:
                        y = common_year
                    ymd_filter = []
                    # date format in database is yyyy/MM/dd HH:mm:00Z
                    if y and m and d:
                        date_string = f"{y}/{m:02d}/{d:02d}"
                        ymd_filter = {"term": {"date": date_string}}
                        self.query_visualisation["DATE"].append(f"{d:02d}/{m:02d}/{y}")
                    elif y and m:
                        date_string = f"{m:02d}/{y}"
                        ymd_filter = {"term": {"month_year": date_string}}
                        self.query_visualisation["DATE"].append(f"{m:02d}/{y}")
                    elif y and d:
                        date_string = f"{d:02d}/{y}"
                        ymd_filter = {"term": {"day_year": date_string}}
                        self.query_visualisation["DATE"].append(f"{d:02d}/-/{y}")
                    elif m and d:
                        date_string = f"{d:02d}/{m:02d}"
                        ymd_filter = {"term": {"day_month": date_string}}
                        self.query_visualisation["DATE"].append(f"{d:02d}/{m:02d}")
                    elif y:
                        ymd_filter = {"term": {"year": y}}
                        self.query_visualisation["DATE"].append(f"{y}")
                    elif m:
                        ymd_filter = {"term": {"month": f"{m:02d}"}}
                        self.query_visualisation["DATE"].append(num2month[m])
                    elif d:
                        ymd_filter = {"term": {"day": f"{d:02d}"}}
                        self.query_visualisation["DATE"].append(f"{d:02d}")

                    self.date_filters.append(ymd_filter)
                # combine the date filters using a bool should clause
                self.date_filters = {"bool": {"should": self.date_filters, "minimum_should_match": 1}}
                
            if self.start[0] != 0 or self.start[1] != 0 or self.end[0] != 24 or self.end[1] != 0:
                self.query_visualisation["TIME"] = [f"{self.start[0]:02d}:{self.start[1]:02d} - {self.end[0]:02d}:{self.end[1]:02d}"]
            
            if self.duration:
                self.duration_filters = [range_filter(self.duration / 2, round(self.duration * 1.5), "duration", 0.1),
                                         range_filter(self.duration / 2, round(self.duration * 1.5), "group_duration", 0.05)]
                self.duration_filters = {"bool": {"should": self.duration_filters, "minimum_should_match": 1}}
                
        return self.time_filters, self.date_filters, self.duration_filters

    def make_location_query(self):
        if not self.location_filters:
            for loc in self.locations:
                place = gps_not_lower[loc]
                place = gps_not_lower[loc]
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
                        # self.location_queries.append({
                        #         "distance_feature": {
                        #             "field": "gps",
                        #             "pivot": pivot,
                        #             "origin": [lon, lat],
                        #             "boost": score * 50
                        #         }
                        #     })
                        self.location_filters.append({
                            "geo_distance": {
                                "distance": dist,
                                "gps": [lon, lat]
                            }
                        })
                        break

            # # General:
            # if len(self.gps_results) < len(self.locations):
            #     for loc in self.locations[len(self.gps_results):]:
            #         loc_set = set(loc.split())
            #         for place, (lat, lon) in map_visualisation:
            #             set_place = gps_location_sets[place]
            #             if loc_set.issubset(set_place):
            #                 pivot = "5m"
            #                 if "airport" in set_place:
            #                     pivot = "200m"
            #                     self.location_filters.append({
            #                         "geo_distance": {
            #                             "distance": "2km",
            #                             "gps": [lon, lat]
            #                         }
            #                     })
            #                 elif "dcu" in set_place:
            #                     pivot = "100m"

            #                 self.location_queries.append({
            #                     "distance_feature": {
            #                         "field": "gps",
            #                         "pivot": pivot,
            #                         "origin": [lon, lat],
            #                         "boost": len(loc_set) / len(set_place) * 50
            #                     }
            #                 })
        # if self.location_queries:
            # return {"dis_max": {"queries": self.location_queries, "tie_breaker": 0.0}}
            if self.location_filters:
                self.location_filters.append({"match": {"location": {"query": " ".join(self.locations), "boost": 0.01}}})
                self.location_filters = {"bool": {"should": self.location_filters, "minimum_should_match": 1}}
        return self.location_filters
