
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

Question: "What is the brand of my car?"
Rewrite: "The logo of my car."

Question: "What is the brand of my glasses?"
Rewrite: "A close-up of my glasses."

Question: "How long did I go running last night? Assume I only went running once."
Rewrite: "I was running last night."

Question: "How many hours did I spend at work last week?"
Rewrite: "I was at work last week."

Question: "How long is my average commute time to work?"
Rewrite: "I was commuting to work."

Question: "What is the color of my car?"
Rewrite: "The outside of my car."

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
