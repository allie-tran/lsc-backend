from autocorrect import Speller
from nltk import pos_tag
from collections import defaultdict
from ..nlp_utils.common import *
from ..nlp_utils.pos_tag import *
from ..nlp_utils.time import *
from ..nlp_utils.synonym import *

init_tagger = Tagger(locations)
e_tag = ElementTagger()


def process_time(time_info):
    weekdays = set()
    dates = set()
    start = (0, 0)
    end = (24, 0)
    for time in time_info:
        if time.info == "WEEKDAY":
            weekdays.add(" ".join(time.name))
        elif time.info == "TIMERANGE":
            s, e = " ".join(time.name).split("-")
            start = adjust_start_end("start", start, *am_pm_to_num(s))
            end = adjust_start_end("end", end, *am_pm_to_num(e))
        elif time.info == "TIME":
            if set(time.prep).intersection(["before", "earlier than", "sooner than"]):
                end = adjust_start_end(
                    "end", end, *am_pm_to_num(" ".join(time.name)))
            elif set(time.prep).intersection(["after", "later than"]):
                start = adjust_start_end(
                    "start", start, *am_pm_to_num(" ".join(time.name)))
            else:
                h, m = am_pm_to_num(" ".join(time.name))
                start = adjust_start_end("start", start, h - 1, m)
                end = adjust_start_end("end", end, h + 1, m)
        elif time.info == "DATE":
            dates.add(get_day_month(" ".join(time.name)))
        elif time.info == "TIMEOFDAY":
            t = time.name[0]
            if "early" in time.prep:
                if "early; " + time.name[0] in timeofday:
                    t = "early; " + time.name[0]
            elif "late" in time.prep:
                if "late; " + time.name[0] in timeofday:
                    t = "late; " + time.name[0]
            if t in timeofday:
                s, e = timeofday[t].split("-")
                start = adjust_start_end("start", start, *am_pm_to_num(s))
                end = adjust_start_end("end", end, *am_pm_to_num(e))
            else:
                print(t, f"is not a registered time of day ({timeofday})")
    return list(weekdays), start, end, list(dates)


def extract_info_from_tag(tag_info):
    objects = set()
    verbs = set()
    locations = set()
    region = set()
    # loc, split_keywords, info, weekday, month, timeofday,
    for action in tag_info['action']:
        if action.name:
            verbs.add(" ".join(action.name))
        if action.in_obj:
            objects.add(" ".join(action.in_obj))
        if action.in_loc:
            locations.add(" ".join(action.in_loc))

    for obj in tag_info['object']:
        for name in obj.name:
            objects.add(name)

    for loc in tag_info['location']:
        for name, info in zip(loc.name, loc.info):
            if info == "REGION":
                region.add(name)
            locations.add(name)

    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}
    objects = objects.difference({""})
    new_objects = set()
    for keyword in objects:
        # if keyword not in all_keywords:
        #     corrected = speller(keyword)
        #     if corrected in all_keywords:
        #         print(keyword, '--->', corrected)
        #         keyword = corrected
        new_objects.add(keyword)
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["descriptions"]["expanded"].append(kw)
    weekdays, start_time, end_time, dates = process_time(tag_info["time"])
    return list(new_objects), split_keywords, list(region), list(locations.difference({""})), list(weekdays), start_time, end_time, list(dates)


def extract_info_from_sentence(sent):
    sent = sent.replace(', ', ',')
    tense_sent = sent.split(',')

    past_sent = ''
    present_sent = ''
    future_sent = ''

    for current_sent in tense_sent:
        split_sent = current_sent.split()
        if split_sent[0] == 'after':
            past_sent += ' '.join(split_sent) + ', '
        elif split_sent[0] == 'then':
            future_sent += ' '.join(split_sent) + ', '
        else:
            present_sent += ' '.join(split_sent) + ', '

    past_sent = past_sent[0:-2]
    present_sent = present_sent[0:-2]
    future_sent = future_sent[0:-2]

    list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(list_sent):
        tags = init_tagger.tag(tense_sent)
        obj = []
        loc = []
        period = []
        time = []
        timeofday = []
        for word, tag in tags:
            if word not in stop_words:
                if tag in ['NN', 'NNS']:
                    obj.append(word)
                if tag in ['SPACE', 'LOCATION']:
                    loc.append(word)
                if tag in ['PERIOD']:
                    period.append(word)
                if tag in ['TIMEOFDAY']:
                    timeofday.append(word)
                if tag in ['TIME', 'DATE', 'WEEKDAY']:
                    time.append(word)
        if idx == 0:
            info['past']['obj'] = obj
            info['past']['loc'] = loc
            info['past']['period'] = period
            info['past']['time'] = time
            info['past']['timeofday'] = timeofday
        if idx == 1:
            info['present']['obj'] = obj
            info['present']['loc'] = loc
            info['present']['period'] = period
            info['present']['time'] = time
            info['present']['timeofday'] = timeofday
        if idx == 2:
            info['future']['obj'] = obj
            info['future']['loc'] = loc
            info['future']['period'] = period
            info['future']['time'] = time
            info['future']['timeofday'] = timeofday

    return info


speller = Speller(lang='en')

class Query:
    def __init__(self, text):
        tags, keywords = init_tagger.tag(text.strip(". \n"))
        self.init_tags = tags
        self.text = [tag[0] for tag in tags]
        self.keywords = [keyword[0] for keyword in keywords]
        self.tags = e_tag.tag(tags)
        self.regions = set()
        self.exacts = set()
        self.expandeds = set()
        self.verbs = set()
        self.objects = set()
        self.locations = set()

#         for action in self.tags['action']:
#             if action.name:
#                 self.verbs.add(str(action))

        for loc in self.tags['location']:
            for name, info in zip(loc.name, loc.info):
                if info == "REGION":
                    self.regions.add(name)
                else:
                    self.locations.add(name)

        for obj in self.tags["object"]:
            for color, name in zip(obj.color, obj.name):
                name = wn.morphy(name)
                self.objects.add((color, name))
                for kw in all_keywords:
                    if kw == name:
                        self.exacts.add(kw)
                    if intersect(kw, name):
                        self.expandeds.add(kw)
        self.weekdays, self.start_time, self.end_time, self.dates = process_time(
            self.tags["time"])
        self.exacts.update(self.keywords)


    def expand(self, must_not_terms=[]):
        self.expansions = defaultdict(lambda: defaultdict(lambda: []))
        musts = self.exacts | self.expandeds

        for word in self.exacts:
            # if 'color' in check_category(word):
                # continue
            possible_words = []
            if word in all_keywords:
                possible_words = [word]
            else:
                for w in to_deeplab(word):
                    possible_words.append(w)

            for w in possible_words:
                if w in all_keywords:
                    self.expandeds.add(w)

            musts.update(possible_words)
            similars = get_similar(word)
            for w in similars:
                self.expansions[word][w.replace('_', ' ')].append(
                    0.99 / similars[w] if similars[w] > 0 else 1)

            for w, dist in get_most_similar(model, word, all_keywords)[:20]:
                self.expansions[word][w.replace('_', ' ')].append(1-dist)

        for keyword in set(self.expandeds):
            for w, dist in get_most_similar(model, keyword, all_keywords)[:20]:
                self.expansions[word][w.replace('_', ' ')].append((1-dist)/2)

        score = defaultdict(lambda: defaultdict(lambda: []))
        for word in self.expansions:
            # if 'color' in check_category(keyword):
                # continue
            for w, dist in self.expansions[word].items():
                if w not in must_not_terms:
                    max_dist = max(dist)
                    musts.add(w)
                    if max_dist > 0.8:
                        score[word][w] = max_dist

        for w in self.exacts:
            # if 'color' in check_category(w):
                # continue
            if w in conceptnet:
                for sym in conceptnet[w]:
                    musts.add(sym)
                    score[w][sym] = 0.99

        for action in self.tags['action']:
            for name in action.name:
                if name in all_keywords:
                    musts.add(name)
                    score[name] = {name: 1}

        for w in score:
            score[w] = dict(sorted(score[w].items(), key=lambda x: -x[1])[:10])

        # TEMPORARY
        temp_scores = {}
        for w in score:
            temp_scores.update(score[w])

        visualise = {}
        for word, tag in self.init_tags:
            if word in visualise:
                continue
            role = ""
            origin_word = wn.morphy(word)
            if not origin_word:
                origin_word = word
            if origin_word in score:
                role = ",".join(score[origin_word].keys())
            elif origin_word in self.exacts:
                role = 'exact'
            elif tag in ["TIME", "DATE", "LOCATION", "REGION", "WEEKDAY", "TIMEPREP", "TIMEOFDAY"]:
                role = tag
            elif tag in ["SPACE"]:
                role = "LOCATION"
            category = check_category(word)
            if 'color' in category:
                role = "COLOR"
            if role:
                visualise[word] = role
        visualise = list(visualise.items())

        musts = musts.difference(["airplane", "plane"])
        return self.exacts, list(musts), list(temp_scores.keys()), temp_scores, visualise



def process_query2(sent):
    tags, keywords = init_tagger.tag(sent)
    tags = e_tag.tag(tags + keywords)
    return extract_info_from_tag(tags)

def process_query3(sent):
    tags = init_tagger.tag(sent)
    timeofday = []
    weekdays = []
    locations = []
    info = []
    activity = []
    month = []
    region = []
    keywords = []
    for word, tag in tags:
        if word == "airport":
            activity.append("airplane")
        if tag == 'TIMEOFDAY':
            timeofday.append(word)
        elif tag == "WEEKDAY":
            weekdays.append(word)
        elif word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                      "november", "december"]:
            month.append(word)
        elif tag == "ACTIVITY":
            if word == "driving":
                activity.append("transport")
                info.append("car")
            elif word == "flight":
                activity.append("airplane")
            else:
                activity.append(word)
            keywords.append(word)
        elif tag == "REGION":
            region.append(word)
        elif tag == "KEYWORDS":
            keywords.append(word)
        elif tag in ['NN', 'SPACE', "VBG", "NNS"]:
            if word in ["office", "meeting"]:
                locations.append("work")
            corrected = speller(word)
            if corrected in all_keywords:
                keywords.append(corrected)
            info.append(word)


    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}
    objects = set(keywords).union(info).difference({""})
    new_objects = set()
    for keyword in objects:
        new_objects.add(keyword)
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if intersect(kw, keyword):
                split_keywords["descriptions"]["expanded"].append(kw)
    return list(new_objects), split_keywords, list(region), [], list(weekdays), (0, 0), (24, 0), []


# tags = init_tagger.tag(
#     "Find the moments in 2015 and 2018 when u1 was using public transports in my home country (Ireland)")
# print(tags)
# tags = e_tag.tag(tags)
# print(tags)
# print(extract_info_from_tag(tags))
