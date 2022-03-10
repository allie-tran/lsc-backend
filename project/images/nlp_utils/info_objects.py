import nltk

from ..nlp_utils.common import flatten_tree_tags, flatten_tree
from pprint import pprint

##### LOCATION TAG #####
class Location:
    def __init__(self, tree_tags):
        # print(tree_tags)
        self.name = []
        self.info = []
        self.prep = ""
        self.extra = ""
        self.tree = tree_tags
        self.extract(tree_tags)
        # self.name = ", ".join(self.name)
        # self.extra = " ".join(self.extra)
        if self.prep == "":
            self.prep = "None"

    def extract(self, t):
        if isinstance(t, nltk.tree.Tree):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["LOCATION", "SPACE", "NN", "NNS", "REGION", "KEYWORD"]:
                self.name.append(t[0])
                self.info.append(t[1])
            elif t[1] in ["PREP", "IN"]:
                self.prep = t[0]
            elif t[1] in ["RB", "PRP$"] and self.extra == "":
                if t[0] == "not" or t[0] != "my":
                    self.extra = "not"
                else:
                    self.extra = ""

    def __repr__(self):
        return f"({self.prep}) <{self.extra}> {self.name}"


##### OBJECT TAG #####
class Object:
    def __init__(self, tree_tags):
        # print(tree_tags)
        self.name = []
        self.position = []
        self.quantity = []
        self.attributes = []
        tree_tags = flatten_tree_tags(tree_tags, ["NN", "KEYWORD", "NNS", "JJ"],[])
        self.extract(tree_tags)
        self.tree = tree_tags
        self.position = " ".join(self.position)
        if not self.position:
            self.position = "None"

    def extract(self, t):
        if not isinstance(t[0], str):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["NN", "NNS", "KEYWORD"]:
                self.name.append(t[0])
                if len(self.quantity) < len(self.name):
                    if self.name[-1][-1] == "s":
                        self.quantity.append("many")
                    else:
                        self.quantity.append("one")
                if len(self.attributes) < len(self.name):
                    self.attributes.append("")
            elif t[1] in ["QUANTITY", "CD"]:
                self.quantity.append(t[0])
            elif t[1] in ["POSITION"]:
                self.position.append(t[0])
            elif t[1] in ["JJ", "ATTRIBUTE"]:
                self.attributes.append(t[0])

    def __repr__(self):
        return f"({self.position}) {'; '.join([f'{a}, {q}, {n}' for a, q, n in zip(self.attributes, self.quantity, self.name)])}"


##### TIME TAG #####
class Time:
    def __init__(self, tree_tags):
        # print(tree_tags)
        self.name = []
        self.period = []
        self.prep = []
        self.info = ""
        tree_tags = flatten_tree_tags(
            tree_tags, ["PERIOD", "TIMEOFDAY",  "TIMERANGE"],[])
        self.extract(tree_tags)

    def extract(self, t):
        if not isinstance(t[0], str):
            [self.extract(l) for l in t]
        else:
            if t[1] in ["TIMEOFDAY", "WEEKDAY", "TIME", "DATE", "TIMERANGE"]:
                self.name.append(t[0])
                self.info = t[1]
            elif t[1] in ["PERIOD", "CD", "NN"]:
                self.period.append(t[0])
            elif t[1] in ["IN", "TIMEPREP", "TO"]:
                try:
                    if self.prep[-1] != 'from' and self.prep[-1] != 'to':
                        self.prep.append(t[0])
                except:
                    self.prep.append(t[0])

    def __repr__(self):
        # mystr = list(map(self.func, self.prep, self.period, self.name))
        # return ", ".join(mystr)
        return "; ".join([self.info] + self.prep + self.period + self.name)


##### ACTION TAG #####
class Action:
    def __init__(self, tree_tags):
        # print(tree_tags)
        self.name = []
        self.in_obj = []
        self.in_loc = []
        self.obj = []
        self.loc = []
        self.time = []
        tree_tags = flatten_tree_tags(
            tree_tags, ["VERB_ING", "PAST_VERB", "VERB", "KEYWORD"], ["OBJECT"])
        self.extract(tree_tags)
        self.calibrate()
        self.tree = tree_tags
        # time, action, object, location
        self.func = lambda x, y, z: " ".join(f"{x} {y} {z}".split())

    def extract(self, t):
        if isinstance(t[0], str):
            if t[1] in ["VERB_ING", "PAST_VERB", "VERB"]:
                self.name.append(t[0])
            elif t[1] in ["NN", "NNS", "KEYWORD"]:  # prior object than location
                self.obj.append(t)
                if (len(self.name) > len(self.in_obj)) == 1:
                    self.in_obj.append(flatten_tree(t[0]))
            elif t[1] in ["LOCATION", "SPACE"]:
                self.loc.append(t)
                if (len(self.name) > len(self.in_loc)):
                    self.in_loc.append(flatten_tree(t[0]))
            elif t[1] in ["TIMEPREP"]:
                if t[0] == 'after':
                    self.time.append('past')
                elif t[0] == 'then' or t[0] == 'before':
                    self.time.append('future')
                else:
                    self.time.append('present')
            # In case there is no TIMEPREP --> default present
            if len(self.time) < len(self.name):
                self.time.append('present')
        elif len(t) ==2 and t[1] == "tree":
            if t[0].label() in ["NN", "NNS", "KEYWORD", "OBJECT"]:  # prior object than location
                self.obj.append(t[0])
                if (len(self.name) > len(self.in_obj)) == 1:
                    self.in_obj.append(flatten_tree(t[0]))
            elif t[0].label() in ["LOCATION", "SPACE"]:
                self.loc.append(t[0])
                if (len(self.name) > len(self.in_loc)):
                    self.in_loc.append(flatten_tree(t[0]))

        else:
            [self.extract(l) for l in t]


    def calibrate(self):
        n_action = len(self.name)
        for i in range(n_action - len(self.time)):
            self.time.append('present')
        for i in range(n_action - len(self.in_obj)):
            self.in_obj.append('')
        for i in range(n_action - len(self.in_loc)):
            self.in_loc.append('')

    def __repr__(self):
        mystr = list(map(self.func, self.name, self.in_obj, self.in_loc))
        return ", ".join(mystr)
