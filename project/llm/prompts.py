"""
Prompts for GPT-4
"""
INSTRUCTIONS = """You are a helpful assistant for a lifelogger who records their daily activities in photos, time and places. Something to help you make the best guess:
 - Assume the lifelogger is Irish and use Irish/British English for dates, times, and word usage.
 - Use self-referential pronouns like "I" and "my" to refer to the lifelogger.
 - When answering questions, call the lifelogger "you".

Always include one single valid JSON object. Do not return multiple JSON objects.
"""

# Answer the question based on text
QA_PROMPT = """Answer the following question based on the text provided: {question}
There are {num_events} retrieved using the lifelog retrieval system. Here are the non-visual information of each event. Bear in mind that some data might be wrong, but the order of relevance is mostly correct (using the system's best guess). Some events might be fragmented and need to be merged to form a complete event.
{events}
Give one or more of your best, educated guesses to the question in the following format, with each answer being the key. The explanation should be brief. You don't have to give a full sentence, just list the reasons. If you can't answer the question, return an empty array for "answers".

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
- top_k: int # number of top results to consider for answering, 0 means none, -1 means all (for "how many", or "how often" questions), default to 10 to consider inaccuracy in the retrieval system

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

