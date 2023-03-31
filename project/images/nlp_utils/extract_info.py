from collections import defaultdict
from ..nlp_utils.common import *
from ..nlp_utils.pos_tag import *
from ..nlp_utils.time import *
# from ..nlp_utils.synonym import *
init_tagger = Tagger([])
time_tagger = TimeTagger()
e_tag = ElementTagger()


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
    text = " " + text + " "
    for keyword in wordset:
        if keyword:
            if " " + keyword + " " in text:
                results.append(keyword)
    return results

class Query:
    def __init__(self, text, shared_filters=None):
        self.negative = ""
        if "NOT" in text:
            text, self.negative = text.split("NOT")
            self.negative = self.negative.strip(". \n").lower()
            self.negative = [word for word in self.negative.split() if word in all_people]
        text = text.strip(". \n").lower()
        self.date_filters = None
        self.time_filters = None
        self.persons = None
        self.folders = None
        self.visits = None
        self.query_visualisation = defaultdict(list)
        self.extract_info(text, shared_filters)

    def extract_info(self, text, shared_filters=None):
        def search_words(wordset):
            return search(wordset, text)
        self.original_text = text

        self.weekdays = []
        self.dates = None
        self.start = (0, 0)
        self.end = (24, 0)
        self.persons = search_words(all_people)
        print(self.persons)
        for person in self.persons:
            text = text.replace(person, "").replace("  ", " ")
        if self.persons:
            self.query_visualisation["PERSON"] = self.persons

        self.folders = search_words(all_dates)
        print(self.folders)
        for folder in self.folders:
            text = text.replace(folder, "").replace("  ", " ")
        if self.folders:
            self.query_visualisation["FOLDER"] = self.folders


        self.visits = search_words(all_visits)
        print(self.visits)
        for visit in self.visits:
            text = text.replace(visit, "").replace("  ", " ")

        changed_visits = []
        suffixes = ["th", "st", "nd", "rd", ] + ["th"] * 16
        for visit in self.visits:
            pattern = re.compile("^visit (\d+)$")
            if pattern.search(visit):
                num = int(pattern.search(visit).group(1))
                visit = str(num) + suffixes[num % 100] + " visit"
            changed_visits.append(visit)
        self.visits = changed_visits

        if self.visits:
            self.query_visualisation["VISIT"] = self.visits


        tags = time_tagger.tag(text)
        processed = set()
        for i, (word, tag) in enumerate(tags):
            if word in processed:
                continue
            if tag in ["WEEKDAY", "TIMERANGE", "TIMEPREP", "DATE", "TIME"]:
                processed.add(word)
                # self.query_visualisation["TIME" if "TIME" in tag else tag].append(word)
            if tag == "WEEKDAY":
                self.weekdays.append(word)
            elif tag == "TIMERANGE":
                s, e = word.split("-")
                self.start = adjust_start_end(
                    "start", self.start, *am_pm_to_num(s))
                self.end = adjust_start_end("end", self.end, *am_pm_to_num(e))
            elif tag == "TIME":
                if word in ["2019", "2020", "2016", "2017"]:
                    self.dates = get_day_month(word)
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
                self.dates = get_day_month(word)
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
        print(processed)
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
        self.clip_text = self.clip_text.strip(", ")
        print("CLIP:", self.clip_text)
        # self.query_visualisation[self.clip_text] = "CLIP"

    def get_info(self):
        return {"query_visualisation": [(hint, ", ".join(value)) for hint, value in self.query_visualisation.items()]}

    def time_to_filters(self):
        if not self.time_filters:
            # Time
            self.time_filters = {
                                    "range":
                                    {
                                        "seconds_from_midnight":
                                        {
                                            "gte": self.start[0] * 3600 + self.start[1] * 60,
                                            "lte": self.end[0] * 3600 + self.end[1] * 60
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
                        {"term": {"day": str(d).rjust(2, "0")}})
            if self.start[0] != 0 or self.end[0] != 24:
                self.query_visualisation["TIME"] = [f"{self.start[0]:02d}:{self.start[1]:02d} - {self.end[0]:02d}:{self.end[1]:02d}"]
            if str(self.dates) != "None":
                self.query_visualisation["DATE"] = [str(self.dates)]
        return self.time_filters, self.date_filters
