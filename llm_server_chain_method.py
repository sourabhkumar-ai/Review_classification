# llm_server.py

from fastapi import FastAPI
from langchain.llms import LlamaCpp
from pydantic import BaseModel, ValidationError
import json
from pydantic import BaseModel, Field
from typing import Annotated, Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# 1. Load the LLM when the server starts
llm = LlamaCpp(
    model_path="/home/bonami/.cache/huggingface/hub/models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/snapshots/3a6fbf4a41a1d52e415a4958cde6856d34b2db93/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    temperature=0.7,
    max_tokens=512,
    top_p=0.95,
    n_ctx=2048,
    verbose=True,
)

app = FastAPI()

# 2. Define the Pydantic model
class ReviewRequest(BaseModel):
    text: str

# class reviews(BaseModel):
#     summary: Annotated[str, "Provide a brief summary about the review"]
#     sentiment: str

# Define the set of possible app components
AppComponent = Literal["UI/UX", "Performance", "Bugs", "Features", "Privacy/Security", "Customer Support"]
SentimentComponent = Literal["Positive","Negative","Neutral"]

class reviews(BaseModel):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[SentimentComponent, "The overall sentiment of the review (positive, negative, or neutral)"]
    segment: Annotated[AppComponent, "The specific part of the app the review is primarily about (e.g., UI, Performance, Bugs)"]



# 3. Create an API endpoint for the review analyzer
@app.post("/analyze_review")   
async def review_analyzer_tool(request: ReviewRequest) -> dict:
    """
    Analyzes a review and returns a structured dictionary
    containing a summary, sentiment, and app segment.
    """
    
    # The rest of the function remains the same, but it will
    # now use the `DetailedReviews` model for validation.
    parser = JsonOutputParser(pydantic_object=reviews)

    template = f"""
    [INST] You are a helpful assistant that analyzes app reviews.
    Analyze the following review and provide a brief summary, an overall sentiment, and classify which segment of the app the review is primarily about.
    
    The sentiment must be one of the following **exact string values**: 'Positive', 'Negative', or 'Neutral'.
    The segment must be one of the following **exact string values**: 'UI/UX', 'Performance', 'Bugs', 'Features', 'Privacy/Security', 'Customer Support', or 'Other'.
    If the review's content does not fit any of the defined segments, use 'Other'.
    
    Return your response as a JSON object with three keys: "summary", "sentiment", and "segment".


    Review to analyze: "{request.text}"
    [/INST]
    """
    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm | parser

    query = request.text
    response = chain.invoke({"text": query})

    # print(response)
    return response


# To run this server, from your terminal:
# uvicorn llm_server:app --reload