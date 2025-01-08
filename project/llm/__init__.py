from configs import USE_GROQ
from .models import gpt_llm_model
from .groq import groq_llm_model
from .internlm import internlm_model

llm_model = groq_llm_model if USE_GROQ else gpt_llm_model

