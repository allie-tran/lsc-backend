"""
Prompts for GPT-4
"""


INSTRUCTIONS = "You are a helpful assistant to help me answer questions in a lifelog dataset. I will give you information and the system constraints and you will suggest me how to do it."

# Rewrite the question into a search query
REWRITE_QUESTION = """
Rewrite the following question into a search query: {question}
It should be a statement in natural language to search for the relevant information in the lifelog retrieval system. It should include all the information from the question, excluding the question phrase
For examples, if the question is "What did I eat for breakfast on 5th May 2019?", the search query could be "I was having breakfast on 5th May 2019".

Answer in this format:
```json
{{
    "text": "A statement that is a search query",
    "visual text": "Exclude this if there is no need for visual information. Otherwise, provide a search query for the images"
}}
```
"""

# Answer the question based on text
QA_PROMPT = """Answer the following question based on the text provided: {question}
There are {num_events} retrieved using the lifelog retrieval system. Here are the non-visual information of each event. Bear in mind that some location data might be wrong, but the order of relevance is mostly correct (using the system's best guess).
{events}
Give one or more of your best guess to the question in the following format, with each answer being the key. The explantion should be brief. You don't have to give a full sentence, just list the reasons.
```json
{{
    "something": "brief explanation for why the answer is `something`",
    "one": "brief explanation for why the answer is `one`",
    "5 days": "brief explanation for why the answer is `5 days`",
    ...
}}
```
"""

VISUAL_PROMPT = """<ImageHere>Answer the following question: {question}
Some extra information (non visual): {extra_info}. But focus on the visual information more. Be brief and concise. If you can't answer the question, just say so."""


# Filter relevant fields from the query
RELEVANT_FIELDS_PROMPT = """My database has this schema:
```
location: str # home, work, a restaurant's name, etc.
location_info: str # type of the semantic_location (restaurant, university, etc.)
region: str # cities, states
country: str

start_time: datetime
end_time: datetime
date: datetime
weekday: str
duration: int
duration_unit: str # minute, hour, day, week, month, year

sortby_time: bool # sort by time

groupby_hour: bool # group by hour
groupby_date: bool # group by day
groupby_week: bool # group by week
groupby_month: bool # group by month

groupby_location: bool # group by location
groupby_city: bool # group by city
groupby_country: bool # group by country

ocr: List[str] # text extracted from images
```

Given this query: "{query}", what are the relevant fields that I should consider? Try to choose the most important ones. If two fields are similar, choose the one that is more specific.
For groupby criteria, consider how far apart two events should be split into two different occasions.

Answer in this format. Don't comment.
```json
["field1", "field2", "field3", "field4", "field5", ...]
```
"""


#  This is still very long!!! TODO!
PARSING_QUERY = """I need to find the answer for this question using my lifelog retrieval system. In my system, a flow of processes is needed:
1. Segmentation: this function takes two parameters: max_time, time_gap, and loc_change, where max_time is the maximum time for each segment, time_gap is the maximum time gap between two segments, and loc_change is the type of location change (semantic_location, city, country, continent). The function returns a list of segments, where each segment is a list of events.
2. Retrieval: this function takes a list of segments and a question. It returns a list of events that are relevant to the question. The function takes the top-K events that are relevant to the question.
3. Extraction: this function takes a list of events and a question. It returns the answer to the question. The function extracts the information from the events to answer the question.
4. Answering: this function takes the answer and the question. It returns the answer to the question.
5. Post-processing: re-organize the events (merge, split, or filter) and the answer to the question. Events with the same answers can be grouped together (if it makes sense).

For example, if the question is "What is my favourite airlines to fly with in 2019?", this is what I'm looking for:
- Segmentation: max_time=1 day, time_gap=1 day, loc_change=city
- Retrieval: query="airlines name on boarding pass or brochure', K=50
- Extraction: metadata=["start_city", "end_city"]
- Answering: needs Visual Question Answering=yes, needs OCR=yes, expected answer type=a name, possible answers=["Delta", "United", "American", "Southwest", "JetBlue"], sort by time=no
- Post-processing: sort=time, group=airlines

Now, the question is "{question}". I need you to define these paramters:
Please provide the following JSON structure:
```
{{
    "segmentation": {{
        "max_time": [a time unit in the following: "year", "month", "week", "day", "hour"],
        "time_gap": [in hours],
        "loc_change": [a location unit,  one of the following: "country", "city", "location_name", "continent"]
    }},
    "retrieval": {{
        "search query": [a search query to find the events],
        "K": [number of events to retrieve and extract answers from]
    }},
    "extraction": {{
        "metadata": [a list of metadata to extract from each event, one of the following: "start_time", "end_time", "semantic_location", "duration", "country", "city", "continent" that might be useful to answer the question],
        "needs Visual Question Answering": [true/false],
        "needs OCR": [true/false],
    }},
    "answering": {{
        "expected answer type": "explanation of what the answer should look like",
        "possible answers": [a list of possible answers]
        }},
    "post-processing": {{
        "group": [a way to group the events, one of the following: any time unit, any location unit, "answer"],
        "sort": [a way to sort the events, one of the following: any time unit, any location_unit, "most_common_answer"]
        "aggregate": [a way to aggregate the events, one of the following: "sum", "average, "max", "min"]
        }}
}}
No explanation is needed. Just provide the JSON structure.
```
"""
