import re
import shelve
from typing import Generator, Iterable, List, Tuple, Union

import nltk

from query_parse.constants import STOP_WORDS
from query_parse.types import RegexInterval, Tags


def find_regex(regex, text, escape=False) -> Generator[RegexInterval, None, None]:
    if "\n" in regex or ("#" in regex and "\\#" not in regex):
        regex = re.compile(regex, re.IGNORECASE | re.VERBOSE)
    else:
        regex = re.compile(regex, re.IGNORECASE)
    for m in regex.finditer(text):
        result = m.group()
        start = m.start()
        while len(result) > 0 and result[0] == " ":
            result = result[1:]
            start += 1
        while len(result) > 0 and result[-1] == " ":
            result = result[:-1]
        yield RegexInterval(start=start, end=start + len(result), text=result)


def flatten_tree(t: Union[str, nltk.tree.Tree]) -> str:
    if isinstance(t, str):
        return t
    return " ".join([l[0] for l in t.leaves()])


def flatten_tree_tags(
    t: Union[str, nltk.tree.Tree], pos_str: List[str], pos_tree: List[str]
) -> Union[str, List]:
    if isinstance(t, nltk.tree.Tree):
        if t.label() in pos_str:
            return [flatten_tree(t), t.label()]
        elif t.label() in pos_tree:
            return [t, "tree"]
        else:
            return [flatten_tree_tags(l, pos_str, pos_tree) for l in t]
    else:
        return t


def is_enabled(location: str, disabled: List[str]) -> bool:
    """
    Check if the location is OK to be used
    """
    if location in ["", "the house", "restaurant"] + disabled:
        return False
    return True


def search_keywords(
    wordset: Iterable[str], text: str, disabled: List[str] = []
) -> List[str]:
    """
    Search for keywords in the text
    """
    results = []
    text = " " + text + " "
    for keyword in wordset:
        if is_enabled(keyword, disabled):
            if re.search(r"\b" + re.escape(keyword) + r"\b", text, re.IGNORECASE):
                results.append(keyword)
    return results


def cache(_func=None, *, file_name=None, separator="_"):
    """
    if file_name is None, just cache it using memory, else save result to file
    """
    if file_name:
        d = shelve.open(file_name)
    else:
        d = {}

    def decorator(func):
        def new_func(*args, **kwargs):
            param = separator.join(
                [str(arg) for arg in args] + [str(v) for v in kwargs.values()]
            )
            if param not in d:
                d[param] = func(*args, **kwargs)
            return d[param]

        return new_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
    """
    Replace the last occurrence of a substring in a string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)


def remove_keywords(query: str, keywords: List[str]) -> str:
    """
    Remove keywords from the query
    """
    for keyword in keywords:
        query = rreplace(query, keyword, "", 1)
    return query


def parse_tags(query: str) -> Tuple[str, dict]:
    """
    Parse tags in the query
    """
    # replace --
    query = query.replace("—", "--")

    words = query.split()

    # possible tags
    tags = {
        "--disable-region": [],
        "--disable-location": [],
        "--disable-time": [],
        "--negative": [],
    }

    # Get all indexes of tags
    all_indexes = [i for i, word in enumerate(words) if word in tags]
    if all_indexes:
        for i, begin_index in enumerate(all_indexes):
            # Find the index of the next tag
            end_index = all_indexes[i + 1] if i + 1 < len(all_indexes) else len(words)

            # Add the arguments of the tag to the list of disabled information
            tag = words[begin_index]
            tags[tag].extend(
                [
                    word.strip()
                    for word in " ".join(words[begin_index + 1 : end_index]).split(",")
                ]
            )

        words = words[: all_indexes[0]]

    # Join the remaining words back into a modified query string
    modified_query = " ".join(words)

    # Example output for demonstration purposes
    result = {
        "disabled_locations": tags["--disable-location"],
        "disabled_regions": tags["--disable-region"],
        "disabled_times": tags["--disable-time"],
        "negative": tags["--negative"],
    }
    print(result)
    return modified_query, result


def strip_stopwords(sentence):
    """
    Remove stopwords from the end of a sentence
    """
    if sentence:
        words = sentence.split()
        for i in reversed(range(len(words))):
            if words[i].lower() in STOP_WORDS:
                words.pop(i)
            else:
                break
        return " ".join(words)
    return ""


def get_visual_text(text: str, unprocessed_words: List[Tags]):
    """
    Get the visual text
    """
    # Get the visual text
    last_non_prep = 0
    visual_text = ""

    # Assert unprocessed_words is not empty
    if not unprocessed_words:
        return visual_text

    # Find the last non-preposition word
    for i in range(len(unprocessed_words) + 1):
        if (
            unprocessed_words[-i][1] not in ["DT", "IN"]
            and unprocessed_words[-i][0] not in STOP_WORDS
        ):
            last_non_prep = -i
            break

    if last_non_prep > 1:
        visual_text = " ".join([word[0] for word in unprocessed_words[:last_non_prep]])
    else:
        visual_text = " ".join([word[0] for word in unprocessed_words])

    # Remove stopwords from the end of the sentence
    visual_text = visual_text.strip(", ?")
    visual_text = strip_stopwords(visual_text)
    visual_text = visual_text.strip(", ?")

    return visual_text
