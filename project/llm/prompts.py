"""
Prompts for GPT-4
"""

INSTRUCTIONS = """You are a helpful assistant for a lifelogger who records their daily activities in photos, time and places. Something to help you make the best guess:
 - Assume the lifelogger is Irish and use Irish/British English for dates, times, and word usage.
 - Use self-referential pronouns like "I" and "my" to refer to the lifelogger.
 - When answering questions, call the lifelogger "you".

Always include one single valid JSON object. Do not return multiple JSON objects.
"""

# Rewrite the question into a search query
REWRITE_QUESTION = """
Rewrite the following question into a statement as a retrieval query: {question}
It should be a statement in natural language to search for the relevant information in the lifelog retrieval system. It should include all the information from the question, excluding the question phrase.

Avoid using "search for", "find", or "retrieve" in the query. Keep it descriptive. Don't replace question words with placeholders like somewhere, something (except for someones). Use the active voice, preferable in the past/past continuous tense. Write it in the first person, as if you are telling a story.

Some examples:
Question: "How many times did I go to the gym last month?"
Rewrite: "I was in the gym last month."

Question: "What did I eat for breakfast on Monday?"
Rewrite: "I was eating breakfast on Monday."

Question: "When was the last time I went to the park, assuming today is 2022-03-01?"
Rewrite: "I was in the park, before 2022-03-01."

Question: "Where did I buy the MacBook Pro in summer 2021?"
Rewrite: "I was buying my MacBook Pro in summer 2021."



Answer in this format:

```json
{{
    "text": "A statement that is a search query",
}}
```
"""

# Automatically parse the query into a different format
PARSE_QUERY = """
I need to parse information from a text query to find relevant information from the lifelog retrieval system. As my parser is rule-based, I need you to provide me with the relevant fields that I should consider. Be as specific as possible, because the parser is not very smart.

The fields are:
```
visual: str # what the lifelogger was doing visually
time: str # time hints, like "morning", "after 3pm", "at sunset"
date: str # date hints, like "last year", "on Christmas", "in 2020", "June 2019"
location: str # place's name, like "at home", "in France", "at the Guinness Storehouse", "in Barcelona"
```

If some location details are not specific (names of places, cities, countries), you should include them in visual field as they can be intepreted visually. Similarly, time information such as sunrise, sunset, or meal times can be included in the visual field. Only include the most specific information in the fields.  Don't say "any" or "unknown" or "unspecified". Just leave the field empty if the information is not available. Each field should be SHORT and CONCISE. Avoid saying "within the time range of 01/01/2019 and 01/07/2020" or "in the past" if the date is not specified.

One last thing, include a "must-not" query if there is any information that should be excluded from the search.

Some examples:
Query: "I was biking in the park near my house in the early morning."
Response:

```json
{{
    "main": {{
        "visual": "biking in the park",
        "time": "early morning",
        "location": "in the park near my house"
    }}
}}
```

Query: "I was having an Irish beer on St. Patrick's Day last year. Assuming this year is 2022."
Response:
```json
{{
    "main": {{
        "visual": "having an Irish beer",
        "date": "17th March 2021"
        "location": ""
    }}
}}
```

Query: "When did I go to a restaurant outside of Ireland and not have a Guinness?"
Response:
```json
{{
    "main": {{
        "visual": "going to a restaurant",
        "location": ""
    }},
    "must_not": {{
        "visual": "having a Guinness"
        "location": "in Ireland"
    }}
}}
```

Query: "It was a very cold winter day in New York City. I walked to the bus at sunrise and went to a conference."
Response:
```json
{{
    "main": {{
        "visual": "very cold winter day, walking at sunrise to the bus",
        "location": "New York City",
        "date": "January, February, December"
        "time": "from 9am to 10am" // because it is at winter so sunrise is late
    }},
    "after": {{
        "visual": "going to a conference",
        "location": "New York City", # probably the same as the main location
        "time": "from 9am to 10am",
    }}
}}
```

Now it's your turn. Use no comments.
Provide the relevant fields for the following query:
Query:{query}
Response:
"""

# Parse before and after information
PARSE_BEFORE_AFTER = """
I need to find the relevant information from the lifelog retrieval system based on the before and after information. The system will retrieve the information based on the before and after information. The system will consider the before information as the query and the after information as the answer.
How many hours before and after each event should I consider to find the relevant information? The time unit is in hours. If you don't need to consider before or after, just put 0.

For example, for the events "I am going to the airport to catch a flight to Paris". The before and after information could be:
```json
{{
    "main_query": "Going to the airport",
    "after": 3-4, # 3-4 hours after the event
    "after_query": "I am at Paris"
}}
```

Example 2: "I was working at a cafe to wait for my friend. After that, we went to the cinema."
```json
{{
    "before_query": "working at a cafe",
    "before": 0-1, # waited for 0-1 hour
    "main_query": "meeting my friend at the cafe",
    "after": 1-2, # went to the cinema 1-2 hours after
    "after_query": "went to the cinema"
}}
```

Example 3: "I had a long walk in the park in the morning because I stayed out late the night before."
```json
{{
    "before": 5-7, # it was the morning so, 7-8am, so last night was 10pm-2am, so 5-10 hours before
    "before_query": "stayed out late at night",
    "main_query": "long walk in the park",
}}
```

These are rational guesses. Now, provide the before and after information for the following query:
Query: {query}
"""

# Answer the question based on text
QA_PROMPT = """Answer the following question based on the text provided: {question}
There are {num_events} retrieved using the lifelog retrieval system. Here are the non-visual information of each event. Bear in mind that some location data might be wrong, but the order of relevance is mostly correct (using the system's best guess).
{events}
Give one or more of your best, educated guesses to the question in the following format, with each answer being the key. Remember the retrieval system is not perfect, so the information might not be 100% accurate. The explanation should be brief. You don't have to give a full sentence, just list the reasons. If you can't answer the question, return an empty array for "answers".

Reminder of the question: {question}

Use the following schema in a valid JSON format:
Answers: List[Answer]
Answer:
- answer: str
- explanation: str
- evidence: List[EventLink]
EventLink:
- event_id: int # the event id, starting from 1

```json
{{
    "answers": [
        {{
            "answer": "something",
            "explanation": "brief explanation for why the answer is `something`",
            "evidence": [1, 2, 3]
        }},
        {{
            "answer": "one",
            "explanation": "brief explanation for why the answer is `one`",
            "evidence": [4, 5, 6]
        }},
        {{
            "answer": "5 days",
            "explanation": "brief explanation for why the answer is `5 days`",
            "evidence": [1, 7, 8]
        }}
    ]
}}
```
"""

MIXED_PROMPTS = """Answer the following question based on the text and images provided: {question}.
Some extra information (non visual): {extra_info}. But focus on the visual information more.
Give one or more of your best guess to the question in the following format, with each answer being the key. The explantion should be brief. You don't have to give a full sentence, just list the reasons.

Use the following schema in a valid JSON format:
Answers: List[Answer]
Answer:
- answer: str
- explanation: str

```json
{{
    "answers": [
        {{
            "answer": "something",
            "explanation": "brief explanation for why the answer is `something`",
        }},
        {{
            "answer": "one",
            "explanation": "brief explanation for why the answer is `one`",
        }},
        {{
            "answer": "5 days",
            "explanation": "brief explanation for why the answer is `5 days`",
        }}
    ]
}}
```
"""

VISUAL_PROMPT = """<ImageHere>Answer the following question: {question}
Some extra information (non visual): {extra_info}. But focus on the visual information more. Be brief and concise. If you can't answer the question, just say so.
Reply with a JSON object in the following format:
```json
{{
    "answer": "Your answer here",
    "explanation": "Brief explanation of your answer"
}}
```
"""

SCHEMA = """My database has this schema:

```
# Valid location fields
place: str # home, work, a restaurant's name, etc. This is the most specific location
place_info: str # type of the semantic_location (restaurant, university, etc.)
city: str # city name, like Dublin, Tokyo, etc.
country: str # country name, like Ireland, Japan, etc.

# Valid time fields
date: str # date of the event
month: str
weekday: str
year: int
duration: str # duration of the event

start_time: datetime # very specific, so unless you have a good reason, use other time fields
end_time: datetime # same as above

# Visual information
ocr: List[str] # text extracted from images
```
"""

# Filter relevant fields from the query
RELEVANT_FIELDS_PROMPT = (
    SCHEMA
    + """

And these are valid gap units:
- time_gap: hour, day, week, month, year
- gps_gap: km, meter

Given this query: "{query}", what are the relevant fields that I should consider? Try to choose the most important ones. If two fields are similar, choose the one that is more specific.
For merge_by criteria, consider how far apart two events should be split into two different occasions. Some examples:

Query: "How many times did I go to the gym last month?"
Merge by: day. Max gap: 3 hours.
Query: "How many days did I spend in Paris last year?"
Merge by: day. Max gap: none
Reason: If the lifelogger spent multiple days in Paris, each day should be considered a separate occasion. There is no need to consider the max gap as it's embedded in the merge_by criteria.

Query: "In which city did I spend the most time in 2019?"
Merge by: city, month. Max gap: 1 week
Reason: Events that happen in the same city within a month are considered the same occasion. If two events are in the same city but 1 week apart, they are considered different occasions.

Query: "What is my car's brand and model?"
Merge by: none. Max gap: none
Sort by: score<desc>, time<desc>

Reminder of the query: {query}

Answer in this format.
```json
{{
    "relevant_fields": ["place", "date", "ocr"],
    "max_gap": {{
                    "time_gap": {{"unit": "hour", "value": 5}},
                    "gps_gap": {{"unit": "km", "value": 1}}
                }},
    "merge_by": ["day", "country"], // or any other relevant fields
    "sort_by": [{{ "field": "score", "order": "desc" }}, {{ "field": "time", "order": "desc" }}]
}}
```
"""
)

ANSWER_MODEL_CHOOSING_PROMPT = """
I have two models here to answer the question.
- A text model that can only read text data, which can include location, time, and other non-visual information. It can also provide aggregated information such as counts, averages, etc.
- A visual model that can read images and text, however very costly to use. It is limited to answering questions about a single event at a time.

Which model(s) should I use to answer the question? You can choose one or both models.
Provide your answer in the following schema:

```
answer_models: Dict[str, AnswerModel]
AnswerModel:
- enabled: bool
- top_k: int # number of top results to consider for answering, 0 means none, -1 means all, default to 10 to consider inaccuracy in the retrieval system

Considering this question: "{question}.
Answer in a valid JSON format:
```json
{{
    "answer_models": {{
        "text": {{"enabled": true, "top_k": 10}},
        "visual": {{"enabled": false, "top_k": 0}}
    }}
}}
```
"""

GRAPH_QUERY = (
    SCHEMA
    + """
Generate a vegalite schema (with encoding, without data) to answer this lifelog question:
{question}

Here is the data:
{data}

Answer in this format:
```json
{{
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "title": "Vega-Lite Schema",
    "description": "Vega-Lite Schema to answer the lifelog question",
    "mark": "bar",
    "encoding": {{
        "x": {{"field": "date", "type": "temporal"}},
        "y": {{"field": "count", "type": "quantitative"}}
    }},
    "data": {{ // example data
        "values": [
            "date": "2022-01-01",
            "count": 10
            ]
    }}
    ... // other fields
}}
```
"""
)
