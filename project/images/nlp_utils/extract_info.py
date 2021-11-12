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

def get_vector(word):
    if word:
        if model.wv.__contains__(word.replace(' ', "_")):
            return model[word.replace(' ', "_")]
        words = [model[word]
                for word in word.split() if model.wv.__contains__(word)]
        if words:
            return np.mean(words, axis=0)
        else:
            wn_word = wn.morphy(word)
            if model.wv.__contains__(wn_word):
                return model[wn_word]
    return None


def get_deeplab_vector(word):
    if word:
        words = [model[word]
                 for word in word.split(';') if model.wv.__contains__(word)]
        if words:
            return np.mean(words, axis=0)
        else:
            wn_word = wn.morphy(word)
            if model.wv.__contains__(wn_word):
                return model[wn_word]
    return None



KEYWORD_VECTORS = {keyword: get_vector(keyword)
                   for keyword in all_keywords_without_attributes}
KEYWORD_VECTORS = {keyword: vector for (keyword, vector) in KEYWORD_VECTORS.items() if vector is not None}
DEEPLAB_VECTORS = {keyword: get_deeplab_vector(keyword)
                   for keyword in map2deeplab}
DEEPLAB_VECTORS = {keyword: vector for (
    keyword, vector) in DEEPLAB_VECTORS.items() if vector is not None}

class Word:
    def __init__(self, word, attribute=""):
        self.word = word
        self.attribute = attribute
        self.modifier = 25 if self.attribute else 50

    def expand(self):
        synonyms = [self.word]
        synsets = wn.synsets(self.word.replace(" ", "_"))
        all_similarities = defaultdict(float)
        if synsets:
            syn = synsets[0]
            synonyms.extend([lemma.name().replace("_", " ")
                             for lemma in syn.lemmas()])
            for name in [name.name() for s in syn.closure(hypo, depth=1) for name in s.lemmas()] + \
                    [name.name() for s in syn.closure(hyper, depth=1) for name in s.lemmas()]:
                name = name.replace("_", " ")
                if name in all_keywords:
                    all_similarities[name] = 1.0 * self.modifier
        deeplab_scores = defaultdict(float)
        for i, word in enumerate(synonyms):
            vector = get_vector(word)
            if vector is not None:
                similarities = [(keyword, 1-cosine(get_vector(word),
                                               KEYWORD_VECTORS[keyword])) for keyword in KEYWORD_VECTORS]
                similarities = sorted([s for s in similarities if s[1] > 0.7],  key=lambda s: -s[1])[:10]
                for sym, score in similarities:
                    all_similarities[sym] = max(
                        all_similarities[sym], score * (1-i*0.1) * self.modifier)
                similarities = [(keyword, 1-cosine(get_vector(word),
                                                   DEEPLAB_VECTORS[keyword])) for keyword in DEEPLAB_VECTORS]
                similarities = sorted(
                    [s for s in similarities if s[1] > 0.7],  key=lambda s: -s[1])[:10]
                for sym, score in similarities:
                    sym = deeplab2simple[sym]
                    deeplab_scores[sym] = max(
                        deeplab_scores[sym], score * (1-i*0.1))

            if word in all_keywords:
                all_similarities[word] = 2.0 * (1-i*0.1) * self.modifier

        deeplab_scores = defaultdict(float)
        attributed_similarities = {}
        if self.attribute:
            all_similarities[self.attribute] = 1.0
            for word, score in all_similarities.items():
                if f"{self.attribute} {word}" in all_keywords:
                    attributed_similarities[f"{self.attribute} {word}"] = score * 2

        return dict(sorted(list(all_similarities.items()) + list(attributed_similarities.items()), key=lambda x: -x[1])), deeplab_scores

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
    def __init__(self, text, shared_filters=None):
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
        self.negative = ""
        if "NOT" in text:
            text, self.negative = text.split("NOT")
            self.negative = self.negative.strip(". \n").lower()
            self.negative = [word for word in self.negative.split() if word in all_keywords]
        text = text.strip(". \n").lower()
        self.time_filters = None
        self.date_filters = None
        self.ocr_queries = []
        self.query_visualisation = {}
        self.country_to_visualise = []
        self.extract_info(text, shared_filters)

    def extract_info(self, text, shared_filters=None):
        def search_words(wordset):
            return search(wordset, text)
        self.original_text = text


        quoted_text = " ".join(re.findall(r'\"(.+?)\"', text))
        text = text.replace(f'"{quoted_text}"', "")

        self.ocr = process_for_ocr(quoted_text.split())
        keywords = search_words(all_keywords_without_attributes)
        self.regions = search_words(regions)

        for reg in self.regions:
            self.query_visualisation[reg] = "REGION"
            for country in countries:
                if reg == country.lower():
                    self.country_to_visualise.append(country)

        self.locations = search_words(locations + ["hotel", "restaurant", "store", "centre", "airport", "station", "cafe"])
        for loc in self.locations:
            self.query_visualisation[loc] = "LOCATION"

        processed = set([w for word in self.regions +
                         self.locations for w in word.split()])

        self.weekdays = []
        self.dates = None
        self.start = (0, 0)
        self.end = (24, 0)

        tags = time_tagger.tag(text)
        print(tags)
        for i, (word, tag) in enumerate(tags):
            if tag in ["WEEKDAY", "TIMERANGE", "TIMEPREP", "DATE", "TIME", "TIMEOFDAY"]:
                processed.update(word.split())
                self.query_visualisation[word] = tag
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
        print("Date:", self.dates)
        self.place_to_visualise = []
        self.phrases = []
        unigrams = [
            word for word in text.split() if word not in stop_words]
        for phrase in bigram_phraser[unigrams]:
            to_take = False
            for word in phrase.split('_'):
                if word not in processed:
                    to_take = True
                    break
            if to_take:
                self.phrases.append(phrase.replace('_', ' '))
            for place in visualisations:
                if word in place.lower().split():
                    self.place_to_visualise.append(place)

        attributed_phrases = []
        taken = set()
        for i, phrase in enumerate(self.phrases[::-1]):
            n = len(self.phrases) - i - 1
            if n in taken:
                continue
            attribute = ""
            if n > 0 and self.phrases[n-1] in attribute_keywords:
                attribute = self.phrases[n-1]
                taken.add(n-1)
            attributed_phrases.append(Word(phrase, attribute))
        self.attributed_phrases = attributed_phrases[::-1]


    def expand(self):
        self.seperate_scores = {}
        self.scores = defaultdict(float)
        self.atfidf = defaultdict(float)
        self.keywords = []
        self.useless = []
        for word in self.attributed_phrases:
            expanded, deeplab = word.expand()
            self.seperate_scores[word.__repr__()] = expanded
            for keyword in expanded:
                self.scores[keyword] = max(
                    self.scores[keyword], expanded[keyword])
            for keyword in deeplab:
                self.atfidf[keyword] = max(
                    self.atfidf[keyword], deeplab[keyword])
            to_visualise = [w for w in expanded if w in all_keywords]
            if word.__repr__() in all_keywords:
                self.keywords.append(word.__repr__())
                self.query_visualisation[word.__repr__()] = "KEYWORD\n" + "\n".join(to_visualise)
            else:
                if expanded:
                    self.query_visualisation[word.__repr__(
                    )] = "NON-KEYWORD\n" + "\n".join(to_visualise)
                else:
                    self.useless.append(word.__repr__())

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
            date_filters = []
            if self.dates:
                y, m, d = self.dates
                if y:
                    date_filters.append({"term": {"year": str(y)}})
                if m:
                    date_filters.append({"term": {"month": str(m).rjust(2, "0")}})
                if d:
                    date_filters.append({"term": {"date": str(d).rjust(2, "0")}})
            self.time_filters, self.date_filters = time_filters, date_filters

        factor = 'begin_time' if scene_group else 'time'

        return [time_filter.replace('DUMMY_FACTOR', factor) for time_filter in self.time_filters], self.date_filters

    def make_ocr_query(self):
        if not self.ocr_queries:
            for ocr_keyword in ocr_keywords:
                if ocr_keyword in self.ocr:
                    self.ocr_queries.append(
                        {"rank_feature": {"field": f"ocr_score.{ocr_keyword}", "boost": 500 * self.ocr[ocr_keyword] / (ocrIDF[ocr_keyword] if ocr_keyword in ocrIDF else 1), "linear": {}}})
        return self.ocr_queries
