from collections import defaultdict
import jellyfish
import calendar
from ..nlp_utils.common import *
from ..nlp_utils.time import *
from datetime import datetime, timedelta
time_tagger = TimeTagger()

def text_similarity(text1, text2):
    return jellyfish.jaro_distance(text1, text2)

def filter_locations(location, disabled):
    if location in ["", "the house", "restaurant"] + disabled:
        return False
    return True

def search(wordset, text, disabled=[]):
    results = []
    text = " " + text + " "
    for keyword in wordset:
        if filter_locations(keyword, disabled):
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                results.append(keyword)
    return results

# Partial match only
def search_possible_location(text, disabled=[]):
    results = []
    for location in locations:
        if filter_locations(location, disabled):
            for i, extra in enumerate(locations[location]):
                if filter_locations(extra, disabled):
                    if re.search(r'\b' + re.escape(extra) + r'\b', text, re.IGNORECASE):
                        if location not in results:
                            results.append(location)
                        break
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

def parse_tags(query):
    # Split the query into individual words
    words = query.split()
    
    # possible tags
    tags = {"--disable-region": [], 
            "--disable-location": [], 
            "--disable-time": [],
            "--negative": []}

    # Get all indexes of tags
    all_indexes = [i for i, word in enumerate(words) if word in tags]
    if all_indexes:
        for i, begin_index in enumerate(all_indexes):
            # Find the index of the next tag
            end_index = all_indexes[i + 1] if i + 1 < len(all_indexes) else len(words)
            
            # Add the arguments of the tag to the list of disabled information
            tag = words[begin_index]
            tags[tag].extend([word.strip() for word in " ".join(words[begin_index + 1 : end_index]).split(",")])
        
            
        words = words[:all_indexes[0]]
    
    # Join the remaining words back into a modified query string
    modified_query = " ".join(words)

    # Example output for demonstration purposes
    result = {
        "disabled_locations": tags["--disable-location"],
        "disabled_regions": tags["--disable-region"],
        "disabled_times": tags["--disable-time"],
        "negative": tags["--negative"]
    }
    print(result)
    return modified_query, result


class Query:
    def __init__(self, text, shared_filters=None):
        self.disable_region = []
        text, self.parsed = parse_tags(text)
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
        # For Elasticsearch
        self.cached = False
        self.scroll_id = ""
        self.es_filters = []
        self.es_should = []
        self.max_score = 1.0
        self.min_score = 0.0

    def extract_info(self, text, shared_filters=None):
        def search_words(wordset, disabled=[]):
            return search(wordset, text, disabled)
        self.original_text = text

        self.locations = search_words(locations, self.parsed["disabled_locations"])
        print("Locations:", self.locations)
        self.place_to_visualise = [gps_not_lower[location] for location in self.locations]
        if self.locations:
            self.query_visualisation["LOCATION"].extend(self.locations)
        else:
            possible_locations = search_possible_location(text, self.parsed["disabled_locations"])
            if possible_locations:
                self.query_visualisation["POSSIBLE LOCATION(S)"].append(", ".join(possible_locations))
                
        self.location_infos = search_words(location_infos, self.parsed["disabled_locations"])
        for loc in self.locations:
            text = rreplace(text, loc, "", 1) #TODO!

        self.regions = search_words(regions, self.parsed["disabled_regions"])
        for reg in self.regions:
            self.query_visualisation["REGION"].append(reg)
            if reg in lowercase_countries:
                country = lowercase_countries[reg]
                self.country_to_visualise.append({"country": country, "geojson": countries[country]})
            if reg in ["korea", "england"]:
                country = reg.title()
                if reg == "korea":
                    self.country_to_visualise.append({"country": country, "geojson": countries["South Korea"]})

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
        print(tags)
        processed = set()
        for i, (word, tag) in enumerate(tags):
            if word in self.parsed["disabled_times"]:
                continue
            if tag in ["WEEKDAY", "TIMERANGE", "DATEPREP", "DATE", "TIME", "TIMEOFDAY", "PERIOD"]:
                if word not in all_in_more_timeofday:
                    processed.add(i)
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
                    if i >= 1 and tags[i-1][1] == 'TIMEPREP':
                        timeprep = tags[i-1][0]
                        processed.add(i-1)
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
            elif tag in ["DATE", "HOLIDAY"]:
                if tag == "DATE": 
                    y, m, d = get_day_month(word)
                elif tag == "HOLIDAY":
                    y, m, d = holiday_text_to_datetime(word)
                dateprep = ""
                if i >= 1 and tags[i-1][1] == 'DATEPREP':
                    dateprep = tags[i-1][0]
                    print(tags[i-1])
                print(word, tags[i-1], dateprep)
                if "first day of" in dateprep:
                    d = 1
                elif "last day of" in dateprep:
                    if y and m:
                        monthrange = calendar.monthrange(y, m)
                        d = monthrange[1]
                    elif m != 2:
                        monthrange = calendar.monthrange(2020, m)
                        d = monthrange[1]
                    else:
                        self.dates.append((y, m, 29))
                elif "day after" in dateprep:
                    original_year = y
                    if not y:
                        y = 2020
                    if m and d:
                        dt_object = datetime(y, m, d)
                        dt_object += timedelta(days=1)
                        if y != dt_object.year and original_year:
                            original_year += 1
                        y, m, d = dt_object.year, dt_object.month, dt_object.day
                    y = original_year
                elif "day before" in dateprep:
                    original_year = y
                    if not y:
                        y = 2020
                    if m and d:
                        dt_object = datetime(y, m, d)
                        dt_object -= timedelta(days=1)
                        if y != dt_object.year and original_year:
                            original_year -= 1
                        y, m, d = dt_object.year, dt_object.month, dt_object.day
                    y = original_year
                self.dates.append((y, m, d))
            elif tag == "SEASON":
                for month in seasons[word]:
                    self.dates.append((None, months.index(month) + 1, None))
            elif tag == "TIMEOFDAY":
                if word not in ["lunch", "breakfast", "dinner", "sunrise", "sunset"]:
                    processed.add(i)
                timeprep = ""
                if i > 1 and tags[i-1][1] == 'TIMEPREP':
                    timeprep = tags[i-1][0]
                    processed.add(i-1)
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
                
        if shared_filters:
            if not self.weekdays:
                self.weekdays.extend(shared_filters.weekdays)
            if self.dates is None:
                self.dates = shared_filters.dates
        unprocessed = [(word, tag) for i, (word, tag) in enumerate(tags) if i not in processed]

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
        # self.clip_text = ". ".join(strip_stopwords(sentence) for sentence in self.clip_text.split("."))
        self.clip_text = self.clip_text.strip(", ?")
        self.clip_text = strip_stopwords(self.clip_text)
        self.clip_text = self.clip_text.strip(", ?")
        if self.clip_text:
            print("CLIP:", self.clip_text)
        else:
            print("NO CLIP")

    def get_info(self):
        return {"query_visualisation": [(hint, value) for hint, value in self.query_visualisation.items()],
                "country_to_visualise": self.country_to_visualise,
                "place_to_visualise": self.place_to_visualise}

    def time_to_filters(self):
        if not self.time_filters:
            # Time
            s = self.start[0] * 3600 + self.start[1] * 60
            e = self.end[0] * 3600 + self.end[1] * 60
            if s == 0 and e == 24 * 3600:
                self.time_filters = None
            else:
                print("Time:", self.start, self.end, s, e)
                if s <= e:
                    # OR (should queries)
                    self.time_filters = [range_filter(s, e, "start_seconds_from_midnight"),
                                        range_filter(s, e, "end_seconds_from_midnight"),
                                        {"bool": {"must": # starts before and ends after
                                            [range_filter(0, s, "start_seconds_from_midnight"),
                                            range_filter(e, 24 * 3600, "end_seconds_from_midnight")]}}]
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
                if date[0]:
                    years.add(date[0])
            if len(years) == 1:
                common_year = years.pop()
                # remove the date with the year-only
                if len(self.dates) > 1:
                    new_dates = []
                    for date in self.dates:
                        y, m, d = date
                        if not m and not d and y:
                            continue
                        new_dates.append(date)
                    self.dates = new_dates
            
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
                        self.query_visualisation["DATE"].append(months[m - 1].capitalize())
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
