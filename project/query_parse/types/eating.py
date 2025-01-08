from typing import List, Optional
from myeachtra.dependencies import CamelCaseModel
from enum import StrEnum


class SocioecologicalLevel(StrEnum):
    INTRAPERSONAL = "Intrapersonal"
    PHYSICAL_ENVIRONMENTAL = "Physical Environmental"
    SOCIO_ENVIRONMENTAL = "Socioenvironmental"

class Mood(StrEnum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

MoodExplanation = {
    Mood.POSITIVE: "Visible smile, laughter, or other positive facial expressions",
    Mood.NEGATIVE: "Visible frown, tears, or other negative facial expressions",
    Mood.NEUTRAL: "No visible facial expressions"
}

class EatingLocationSetting(StrEnum):
    GREEN_SPACE = "Green Space"
    HOME = "Home"
    WORKPLACE = "Workplace"
    UNIVERSITY = "University/School"
    FOOD_VENUE = "Food Venue"
    SPORT_VENUE = "Sporting Venue"
    OUT_OF_HOME = "Out of Home"

EatingLocationSettingExplanation = {
    EatingLocationSetting.GREEN_SPACE: "In a green space (park, garden, etc.)",
    EatingLocationSetting.HOME: "At home",
    EatingLocationSetting.WORKPLACE: "At the workplace",
    EatingLocationSetting.UNIVERSITY: "At a university or school",
    EatingLocationSetting.FOOD_VENUE: "At a food venue (restaurant, cafe, etc.)",
    EatingLocationSetting.SPORT_VENUE: "At a sporting venue (stadium, gym, etc.)",
    EatingLocationSetting.OUT_OF_HOME: "Out of home (street, public transport, etc.)"
}

class EatingLocationPosition(StrEnum):
    SITTING_AT_TABLE = "Sitting at a table"
    SITTING_ON_A_CHAIR = "Sitting on a chair"
    SITTING_ON_THE_COUCH = "Sitting on the couch"
    STANDING = "Standing/Walking"

EatingLocationPositionExplanation = {
    EatingLocationPosition.SITTING_AT_TABLE: "Sitting at a table",
    EatingLocationPosition.SITTING_ON_A_CHAIR: "Sitting on a chair",
    EatingLocationPosition.SITTING_ON_THE_COUCH: "Sitting on the couch",
    EatingLocationPosition.STANDING: "Standing or walking"
}

class SocialContact(StrEnum):
    ALONE = "Alone"
    PERSON_PRESENT = "Person(s) present"
    INTERACTING = "Interacting with person(s)"
    GATHERING = "Social event or gathering"

SocialContactExplanation = {
    SocialContact.ALONE: "Alone",
    SocialContact.PERSON_PRESENT: "Person(s) present but not interacting",
    SocialContact.INTERACTING: "Interacting with person(s) (talking, eating together)",
    SocialContact.GATHERING: "Social event or gathering (party, dinner, etc.)"
}

class EatingActivity(StrEnum):
    PHONE = "Using a handheld device (phone, tablet)"
    SCREEN = "Using a screen for work/recreation (computer, console)"
    TV = "Watching TV or a film"
    TRANSIT = "In transit (car, bus, train)"
    RECREATION = "Other recreational activity (reading, listening to music, playing a board game"

class TypeOfFood(StrEnum):
    CORE = "Core food"
    NON_CORE = "Non-core food"
    ALCOHOL = "Alcohol"
    UNCLASSIFIABLE = "Unclassifiable"

class FoodGroup(StrEnum):
    FRUIT = "Fruit"
    VEGETABLE = "Vegetable"
    GRAIN = "Grain"
    PROTEIN = "Meats, fish, eggs, legumes, nuts"
    DAIRY = "Dairy"
    UNHEALTHY = "All unhealthy foods (soft drinks, chips, sweets, etc.)"
    ALCOHOL = "Alcohol beverages"
    NOT_EATING = "Not eating"

class Food(CamelCaseModel):
    name: FoodGroup
    food_group: FoodGroup
    type_of_food: TypeOfFood

class EatingContext(CamelCaseModel):
    mood: Optional[Mood] = None
    location_setting: Optional[EatingLocationSetting] = None
    location_position: Optional[EatingLocationPosition] = None
    social_contact: Optional[SocialContact] = None
    eating_activity: Optional[EatingActivity] = None
    food: List[FoodGroup] = []

