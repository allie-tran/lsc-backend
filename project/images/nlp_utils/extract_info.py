from nltk import pos_tag

from ..nlp_utils.common import *
from ..nlp_utils.pos_tag import *

init_tagger = Tagger(locations)
e_tag = ElementTagger()


def extract_info_from_tag(tag_info):
    obj_dict = {}
    obj_past = []
    obj_present = []
    obj_future = []

    loc_dict = {}
    loc_past = []
    loc_present = []
    loc_future = []

    for action in tag_info['action']:
        extract_action = action.split(';')
        action_time = extract_action[0]
        if action_time == 'past':
            try:
                if extract_action[2] != '':
                    obj_past.append(extract_action[2])
                if extract_action[3] != '':
                    loc_past.append(extract_action[3])
            except:
                pass
        if action_time == 'present':
            try:
                if extract_action[2] != '':
                    obj_present.append(extract_action[2])
                if extract_action[3] != '':
                    loc_present.append(extract_action[3])
            except:
                pass
        if action_time == 'future':
            try:
                if extract_action[2] != '':
                    obj_future.append(extract_action[2])
                if extract_action[3] != '':
                    loc_future.append(extract_action[3])
            except:
                pass

    for obj in tag_info['object']:
        extract_object = obj.split(', ')
        obj_is = extract_object[1]
        if obj_is not in obj_past and obj_is not in obj_present and obj_is not in obj_future:
            if len(obj_past) > 0:
                obj_past.append(obj_is)
            if len(obj_present) > 0:
                obj_present.append(obj_is)
            if len(obj_future) > 0:
                obj_future.append(obj_is)

    for loc in tag_info['location']:
        extract_loc = loc.split()
        loc_is = extract_loc[2]
        negative = True if len(extract_loc[1]) > 2 else False
        if negative:  # since it is NOT --> dont append it
            try:
                loc_past.remove(loc_is)
                loc_present.remove(loc_is)
                loc_future.remove(loc_is)
            except:
                pass
        else:
            if loc_is not in loc_past and loc_is not in loc_present and loc_is not in loc_future:
                if len(loc_past) > 0:
                    loc_past.append(loc_is)
                if len(loc_present) > 0:
                    loc_present.append(loc_is)
                if len(loc_future) > 0:
                    loc_future.append(loc_is)

    query_dict = {}
    query_dict['present'] = obj_present + loc_present
    query_dict['past'] = obj_past + loc_past
    query_dict['future'] = obj_future + loc_future

    obj_dict['present'] = obj_present
    obj_dict['past'] = obj_past
    obj_dict['future'] = obj_future

    loc_dict['present'] = loc_present
    loc_dict['past'] = loc_past
    loc_dict['future'] = loc_future

    return query_dict, obj_dict, loc_dict


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


def extract_info_from_sentence_full_tag(sent):
    # sent = sent.replace(', ', ',')
    # tense_sent = sent.split(';')
    #
    # past_sent = ''
    # present_sent = ''
    # future_sent = ''
    #
    # for current_sent in tense_sent:
    #     split_sent = current_sent.split()
    #     if split_sent[0] == 'after':
    #         past_sent += ' '.join(split_sent) + ', '
    #     elif split_sent[0] == 'then':
    #         future_sent += ' '.join(split_sent) + ', '
    #     else:
    #         present_sent += ' '.join(split_sent) + ', '
    #
    # past_sent = past_sent[0:-2]
    # present_sent = present_sent[0:-2]
    # future_sent = future_sent[0:-2]
    #
    # list_sent = [past_sent, present_sent, future_sent]

    info = {}
    info['past'] = {}
    info['present'] = {}
    info['future'] = {}

    for idx, tense_sent in enumerate(["", sent]):
        if len(tense_sent) > 2:
            tags = init_tagger.tag(tense_sent)
            print(tags)
            info_full = e_tag.tag(tags)
            obj = []
            loc = []
            period = []
            time = []
            timeofday = []

            if len(info_full['object']) != 0:
                for each_obj in info_full['object']:
                    split_term = each_obj.split(', ')
                    if len(split_term) == 2:
                        obj.append(split_term[1])

            if len(info_full['period']) != 0:
                for each_period in info_full['period']:
                    if each_period not in ['after', 'before', 'then', 'prior to']:
                        period.append(each_period)

            if len(info_full['location']) != 0:
                for each_loc in info_full['location']:
                    split_term = each_loc.split('> ')
                    if split_term[0][-3:] != 'not':
                        word_tag = pos_tag(split_term[1].split())
                        final_loc = []
                        for word, tag in word_tag:
                            if tag not in ['DT']:
                                final_loc.append(word)
                        final_loc = ' '.join(final_loc)
                        loc.append(final_loc)

            if len(info_full['time']) != 0:
                for each_time in info_full['time']:
                    if 'from' in each_time or 'to' in each_time:
                        timeofday.append(each_time)
                    else:
                        timetag = init_tagger.time_tagger.tag(each_time)
                        if timetag[-1][1] in ['TIME', 'TIMEOFDAY']:
                            timeofday.append(each_time)
                        elif timetag[-1][1] in ['WEEKDAY', 'DATE']:
                            time.append(timetag[-1][0])

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


def process_query(sent):
    must_not = re.findall(r"-\S+", sent)
    must_not_terms = []
    for word in must_not:
        sent = sent.replace(word, '')
        must_not_terms.append(word.strip('-'))

    tags = init_tagger.tag(sent)
    timeofday = []
    weekday = []
    loc = []
    info = []
    activity = []
    month = []
    region = []
    keywords = []
    print(tags)
    for word, tag in tags:
        if word == "airport":
            activity.append("airplane")
        # if word == "candle":
            # keywords.append("lamp")
        if tag == 'TIMEOFDAY':
            timeofday.append(word)
        elif tag == "WEEKDAY":
            weekday.append(word)
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
                loc.append("work")
            info.append(word)
    print(f"Location: {loc}, weekday: {weekday}, month: {month}, timeofday: {timeofday}, activity: {activity}, region: {region}, must-not: {must_not_terms}")
    print(f"Keywords:", keywords, "Rest:", info)

    split_keywords = {"descriptions": {"exact": [], "expanded": []},
                      "coco": {"exact": [], "expanded": []},
                      "microsoft": {"exact": [], "expanded": []}}

    for keyword in keywords:
        for kw in microsoft:
            if kw == keyword:
                split_keywords["microsoft"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["microsoft"]["expanded"].append(kw)
        for kw in coco:
            if kw == keyword:
                split_keywords["coco"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["coco"]["expanded"].append(kw)
        for kw in all_keywords:
            if kw == keyword:
                split_keywords["descriptions"]["exact"].append(kw)
            if kw in keyword or keyword in kw:
                split_keywords["descriptions"]["expanded"].append(kw)

    return loc, split_keywords, info, weekday, month, timeofday, list(set(activity)), list(set(region)), must_not_terms
