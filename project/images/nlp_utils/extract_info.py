from gensim.models.phrases import Phraser
from gensim.models import Word2Vec
from autocorrect import Speller
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


def morphy(word):
    result = ""
    try:
        result = wn.synsets(word)[0].lemmas()[0].name()
    except IndexError:
        result = wn.morphy(word)
    return result


def process_for_ocr(text):
    final_text = defaultdict(int)
    for word in text:
        final_text[word] += 1
        for i in range(0, len(word)-1):
            if len(word[:i+1]) > 1:
                final_text[word[:i+1]] += (i+1) / len(word)
            if len(word[i+1:]) > 1:
                final_text[word[i+1:]] += 1 - (i+1)/len(word)
    print(final_text)
    return final_text


class Query_old:
    def __init__(self, text):
        self.ocr = " ".join(re.findall(r'\"(.+?)\"', text)).split()
        tags, keywords = init_tagger.tag(text.strip(". \n"))
        self.init_tags = tags
        print(self.init_tags)
        self.text = [tag[0] for tag in tags]
        self.keywords = [keyword[0] for keyword in keywords]
        print(self.text, self.keywords)
        self.tags = e_tag.tag(tags)

        self.regions = set()
        self.exacts = set()
        self.expandeds = set()
        self.verbs = set()
        self.objects = {}
        self.locations = set()

        for loc in self.tags['location']:
            for name, info in zip(loc.name, loc.info):
                if info == "REGION":
                    self.regions.add(name)
                else:
                    self.locations.add(name)

        for word, tag in self.init_tags:
            if tag == "REGION":
                self.regions.add(word)

        for obj in self.tags["object"]:
            for attributes, name in zip(obj.attributes, obj.name):
                origin_word = morphy(name)
                if not origin_word:
                    origin_word = name
                if origin_word in attribute_keywords:
                    continue
                self.objects[origin_word] = attributes
                for kw in all_keywords:
                    if kw == origin_word:
                        self.exacts.add(kw)
                    elif intersect(kw, origin_word):
                        self.expandeds.add(kw)
        self.exacts.update(self.keywords)
        self.weekdays, self.start_time, self.end_time, self.dates = process_time(
            self.tags["time"])

    def expand(self, must_not_terms=[]):
        self.expansions = defaultdict(lambda: defaultdict(lambda: []))
        musts = self.exacts | self.expandeds

        for word in self.exacts:
            if word in attribute_keywords:
                continue
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
            if keyword in attribute_keywords:
                continue
            for w, dist in get_most_similar(model, keyword, all_keywords)[:20]:
                self.expansions[word][w.replace('_', ' ')].append((1-dist)/2)

        score = defaultdict(lambda: defaultdict(lambda: []))

        for word in self.expansions:
            if word in attribute_keywords:
                continue
            for w, dist in self.expansions[word].items():
                if w not in must_not_terms:
                    max_dist = max(dist)
                    musts.add(w)
                    if max_dist > 0.8:
                        score[word][w] = max_dist

        for w in self.exacts:
            if w in attribute_keywords:
                continue
            if w in conceptnet:
                for sym in conceptnet[w]:
                    musts.add(sym)
                    score[w][sym] = 0.99

        for action in self.tags['action']:
            for word in action.name:
                if word in attribute_keywords:
                    continue
                for w, dist in self.expansions[word].items():
                    if w not in must_not_terms:
                        max_dist = max(dist)
                        musts.add(w)
                        if max_dist > 0.8:
                            score[word][w] = max_dist

        for w in score:
            score[w] = dict(sorted(score[w].items(), key=lambda x: -x[1]))
            if w in self.objects:
                new = {}
                for expanded in score[w]:
                    attributed_expanded = f"{self.objects[w]} {expanded}"
                    if attributed_expanded in all_keywords:
                        new[attributed_expanded] = score[w][expanded]
                score[w].update(new)
                score[w] = dict(
                    sorted(score[w].items(), key=lambda x: -x[1]))
        # TEMPORARY
        temp_scores = {}
        for w in score:
            temp_scores.update(score[w])

        musts = musts.difference(["airplane", "plane"])
        extras = set()
        for word in self.exacts:
            if word in self.objects:
                attributed_word = f"{self.objects[word]} {word}"
                if attributed_word in all_keywords:
                    extras.add(attributed_word)
        self.exacts |= extras
        musts |= extras
        visualise = {}
        for word, tag in self.init_tags:
            if word in visualise or word in stop_words:
                continue
            role = ""
            origin_word = word
            if word not in all_keywords:
                origin_word = morphy(word)
                if not origin_word:
                    origin_word = word
            if origin_word in score:
                role = ",".join(score[origin_word].keys())
            elif tag in ["TIME", "DATE", "LOCATION", "REGION", "WEEKDAY", "TIMEPREP", "TIMEOFDAY", "ATTRIBUTE"]:
                role = tag
            elif tag in ["SPACE"]:
                role = "LOCATION"
            elif origin_word in self.exacts:
                role = 'exact'
            if role:
                visualise[word] = role
            if origin_word in score:
                role = ",".join(score[origin_word].keys())
            elif tag in ["TIME", "DATE", "LOCATION", "REGION", "WEEKDAY", "TIMEPREP", "TIMEOFDAY", "ATTRIBUTE"]:
                role = tag
            elif tag in ["SPACE"]:
                role = "LOCATION"
            elif origin_word in self.exacts:
                role = 'KEYWORD'
            if role:
                visualise[word] = role
        visualise = list(visualise.items())
        return self.exacts, list(musts), list(temp_scores.keys()), temp_scores, visualise


@cache
def get_vector(word):
    if word.replace(' ', "_") in model:
        return model[word.replace(' ', "_")]
    words = [model[word] for word in word.split() if word in model]
    if words:
        return np.mean(words, axis=0)
    else:
        return np.zeros(32)


KEYWORD_VECTORS = {keyword: get_vector(keyword)
                   for keyword in all_keywords_without_attributes}


class Word:
    def __init__(self, word, attribute=""):
        self.word = word
        self.attribute = attribute

    def expand(self):
        synonyms = [self.word]
        synsets = wn.synsets(self.word.replace(" ", "_"))
        all_similarities = []
        if synsets:
            syn = synsets[0]
            synonyms.extend([lemma.name().replace("_", " ")
                             for lemma in syn.lemmas()])
            for name in [name.name() for s in syn.closure(hypo, depth=1) for name in s.lemmas()] + \
                    [name.name() for s in syn.closure(hyper, depth=1) for name in s.lemmas()]:
                name = name.replace("_", " ")
                if name in all_keywords:
                    all_similarities.append((name, 1))
        for word in synonyms:
            similarities = [(keyword, 1-cosine(get_vector(word), KEYWORD_VECTORS[keyword])) for keyword in KEYWORD_VECTORS]
            similarities = sorted([s for s in similarities if s[1] > 0.7],  key=lambda s: -s[1])[:10]
            all_similarities.extend(similarities)

        attributed_similarities = []
        if self.attribute:
            for word, dist in similarities:
                if f"{self.attribute} {word}" in all_keywords:
                    attributed_similarities.append(
                        (f"{self.attribute} {word}", dist * 2))
        return dict(sorted(similarities + attributed_similarities, key=lambda x: -x[1]))

    def __repr__(self):
        return " ".join(([self.attribute] if self.attribute else []) + [self.word])


def search(wordset, text):
    results = []
    for keyword in wordset:
        if keyword:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                results.append(keyword)
    return results


def time_es_query(prep, hour, minute, scene_group=False):
    factor = 'end_time' if scene_group else 'time'
    if prep in ["before", "earlier than", "sooner than"]:
        if hour != 24 or minute != 0:
            return f"(doc['{factor}'].value.getHour() < {hour} || (doc['{factor}'].value.getHour() == {hour} && doc['{factor}'].value.getMinute() <= {minute}))"
        else:
            return None
    factor = 'begin_time' if scene_group else 'time'
    if prep in ["after", "later than"]:
        if hour != 0 or minute != 0:
            return f"(doc['{factor}'].value.getHour() > {hour} || (doc['{factor}'].value.getHour() == {hour} && doc['{factor}'].value.getMinute() >= {minute}))"
        else:
            return None
    if scene_group:
        f"(abs(doc['begin_time'].value.getHour() - {hour}) < 1 || abs(doc['end_time'].value.getHour() - {hour}) < 1)"
    return f"abs(doc['time'].value.getHour() - {hour}) < 1"


def add_time_query(time_filters, prep, time, scene_group=False):
    query = time_es_query(prep, time[0], time[1], scene_group)
    if query:
        time_filters.add(query)
    return time_filters


class Query:
    def __init__(self, text):
        # query_ocr = query_info["query_ocr"]
        # query_text = query_info["query_text"]
        # exact_terms = query_info["exact_terms"]
        # must_terms = query_info["must_terms"]
        # must_not_terms = query_info["must_not_terms"]
        # expansion = query_info["expansion"]
        # expansion_score = query_info["expansion_score"],
        # weekdays = query_info["weekdays"]
        # start_time = query_info["start_time"]
        # end_time = query_info["end_time"]
        # dates = query_info["dates"]
        # region = query_info["region"]
        # location = query_info["location"]
        # query_visualisation = query_info["query_visualisation"]
        text = text.strip(". \n")
        self.text = text
        self.time_filters = None
        self.date_filters = None
        self.ocr_queries = []
        self.query_visualisation = {}
        self.country_to_visualise = []
        self.extract_info(text)

    def extract_info(self, text):
        def search_words(wordset):
            return search(wordset, text)

        self.ocr = process_for_ocr(
            " ".join(re.findall(r'\"(.+?)\"', text)).split())
        self.original_text = text
        keywords = search_words(all_keywords_without_attributes)
        self.regions = search_words(regions)
        for reg in self.regions:
            self.query_visualisation[reg] = "REGION"
            for country in countries:
                if reg == country.lower():
                    self.country_to_visualise.append(country)

        self.locations = search_words(locations)
        for loc in self.locations:
            self.query_visualisation[loc] = "LOCATION"
        processed = set()
        processed.update(self.regions + self.locations)

        self.weekdays = set()
        self.dates = set()
        self.start = (0, 0)
        self.end = (24, 0)

        tags = time_tagger.tag(text)
        for i, (word, tag) in enumerate(tags):
            if tag in ["WEEKDAY", "TIMERANGE", "TIMEPREP", "DATE", "TIME", "TIMEOFDAY"]:
                processed.update(word.split())
                self.query_visualisation[word] = tag
            if tag == "WEEKDAY":
                self.weekdays.add(word)
            elif tag == "TIMERANGE":
                s, e = " ".join(word).split("-")
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
                self.dates.add(get_day_month(word))
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

        self.place_to_visualise = []
        phrases = []
        self.unigrams = [
            word for word in text.split() if word not in stop_words]
        for phrase in bigram_phraser[self.unigrams]:
            to_take = False
            for word in phrase.split('_'):
                if word not in processed:
                    to_take = True
                    break
            if to_take:
                phrases.append(phrase.replace('_', ' '))
            for place in visualisations:
                if word in place.lower().split():
                    self.place_to_visualise.append(place)

        attributed_phrases = []
        taken = set()
        for i, phrase in enumerate(phrases[::-1]):
            n = len(phrases) - i - 1
            if n in taken:
                continue
            attribute = ""
            if n > 0 and phrases[n-1] in attribute_keywords:
                attribute = phrases[n-1]
                taken.add(n-1)
            attributed_phrases.append(Word(phrase, attribute))
        self.attributed_phrases = attributed_phrases[::-1]

    def expand(self):
        self.scores = defaultdict(float)
        self.keywords = []
        for word in self.attributed_phrases:
            expanded = word.expand()
            for keyword in expanded:
                self.scores[keyword] = max(
                    self.scores[keyword], expanded[keyword])
            to_visualise = [w for w in expanded][:15]
            if word.__repr__() in all_keywords:
                self.keywords.append(word.__repr__())
                self.query_visualisation[word.__repr__()] = "KEYWORD\n" + "\n".join(to_visualise)
            else:
                if expanded:
                    self.query_visualisation[word.__repr__()] = "\n".join(to_visualise)

        for word in self.keywords:
            self.scores[word] += 20

    def get_info(self):
        return {"query_visualisation": list(self.query_visualisation.items()),
                "country_to_visualise": self.country_to_visualise,
                "place_to_visualise": self.place_to_visualise}

    def time_to_filters(self, scene_group=False):
        if not self.time_filters or not self.date_filters:
            # Time
            time_filters = add_time_query(
                set(), "after", self.start, scene_group=scene_group)
            time_filters = add_time_query(
                time_filters, "before", self.end, scene_group=scene_group)
            if (self.end[0] < self.start[0]) or (self.end[0] == self.start[0] and self.end[1] < self.start[1]):
                time_filters = [
                    f' ({"||".join(time_filters)}) '] if time_filters else []
            else:
                time_filters = [
                    f' ({"&&".join(time_filters)}) '] if time_filters else []
            # Date
            date_filters = set()
            for y, m, d in self.dates:
                this_filter = []
                if y:
                    this_filter.append(
                        f" (doc['DUMMY_FACTOR'].value.getYear() == {y}) ")
                if m:
                    this_filter.append(
                        f" (doc['DUMMY_FACTOR'].value.getMonthValue() == {m}) ")
                if d:
                    this_filter.append(
                        f" (doc['DUMMY_FACTOR'].value.getDayOfMonth() == {d}) ")
                date_filters.add(f' ({"&&".join(this_filter)}) ')
            date_filters = [
                f' ({"||".join(date_filters)}) '] if date_filters else []
            self.time_filters, self.date_filters = time_filters, date_filters

        factor = 'begin_time' if scene_group else 'time'

        return [time_filter.replace('DUMMY_FACTOR', factor) for time_filter in self.time_filters], \
               [date_filter.replace('DUMMY_FACTOR', factor)
                for date_filter in self.date_filters]

    def make_ocr_query(self):
        if not self.ocr_queries:
            for ocr_keyword in ocr_keywords:
                if ocr_keyword in self.ocr:
                    self.ocr_queries.append(
                        {"rank_feature": {"field": f"ocr_score.{ocr_keyword}", "boost": 500 * self.ocr[ocr_keyword], "linear": {}}})
        return self.ocr_queries
