# llm_server.py
import os
import traceback
from dotenv import load_dotenv

from fastapi import FastAPI

from pydantic import BaseModel, ValidationError
from pydantic import BaseModel, Field

from typing import Annotated, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import OpenAI
from langchain.chat_models import init_chat_model
from langchain.prompts.chat import (
                                        ChatPromptTemplate,
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                    )

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

app = FastAPI()

# Define the set of possible app components
AppComponent = Literal["UI/UX", "Performance", "Bugs", "Features", "Privacy/Security", "Customer Support"]
SentimentComponent = Literal["Positive","Negative","Neutral"]

# Define the Pydantic model for the incoming reviews and the output of reviews which is helpful in getting a structured data.
class ReviewRequest(BaseModel):
    text: str

class reviews(BaseModel):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[SentimentComponent, "The overall sentiment of the review (positive, negative, or neutral)"]
    segment: Annotated[AppComponent, "The specific part of the app the review is primarily about (e.g., UI, Performance, Bugs)"]

# API endpoint for the review analysis.
@app.post("/analyze_review")   
async def review_analyzer_tool(request: ReviewRequest) -> dict:
    """
    Analyzes a review and returns a structured dictionary
    containing a summary, sentiment, and app segment(from the defined categories).
    Payload : review -> str
    Reponse : {'summary': 'The reviewer uninstalled the app due to frequent pop-up ads in the free version and vague privacy policy, resulting in discomfort with the amount of personal data collected.', 
    'sentiment': 'Negative', 
    'segment': 'Privacy/Security'}

    """

    # Model initialization : 
    try:
        llm = init_chat_model("gpt-4", 
                            model_provider="openai",temperature=0.7,
                            max_tokens=512,
                            top_p=0.95,
                            )
    except ValueError as e:
        print(f"ValueError during model init: {e}")
        llm = None
        error_message = "Invalid parameters for model initialization."

    except ConnectionError as e:
        print(f"ConnectionError: {e}")
        llm = None
        error_message = "Failed to connect to model provider."

    if llm is None:
        raise RuntimeError(f"Model initialization failed: {error_message}")
    
    parser = JsonOutputParser(pydantic_object=reviews)
    
    # Prompt Creation.
    chat_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a helpful assistant that analyzes app reviews.
                        Analyze the following review and provide a brief summary, an overall sentiment, and classify which segment of the app the review is primarily about.

                        The sentiment must be one of the following **exact string values**: 'Positive', 'Negative', or 'Neutral'.
                        The segment must be one of the following **exact string values**: 'UI/UX', 'Performance', 'Bugs', 'Features', 'Privacy/Security', 'Customer Support', or 'Other'.
                        If the review's content does not fit any of the defined segments, use 'Other'.

                        Return your response as a JSON object with three keys: "summary", "sentiment", and "segment".
                        """
                    ),
                    HumanMessagePromptTemplate.from_template("{text}"),
                ])
    # Chain building.
    try:
        if not hasattr(request, 'text') or not isinstance(request.text, str):
            raise ValueError("Invalid or missing 'text' in request")
        chain = chat_prompt | llm | parser
        raw_response = await chain.ainvoke({"text": request.text})
        return raw_response
    
    except AttributeError as e:
        print(f"AttributeError: {e}")
        return {"error": "Invalid request: missing 'text'"}
    
    except ValueError as e:
        print(f"Value Error {e}")
        return {"error":"Invalid request:Missing text"}
    except TypeError as e:
        print(f"TypeError in chain: {e}")
        return {"error": "Type error during chain execution"}

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        return {"error": "Runtime error during chain execution"}
    
    except TimeoutError as e:
        print(f"TimeoutError: {e}")
        return {"error": "Timeout during model response"}

    except ConnectionError as e:
        print(f"ConnectionError: {e}")
        return {"error": "Failed to connect to model API"}

    except Exception as e:
        
        trace = traceback.format_exc()

        if "openai" in str(type(e)).lower():
            return {"error": "OpenAI API Error", "details": str(e)}
        elif "langchain" in str(type(e)).lower():
            return {"error": "LangChain Error", "details": str(e)}
        else:
            print(f"Unexpected Error: {trace}")
            return {"error": "Unexpected error", "details": str(e)}
    
    

# To run this server, from your terminal:
# uvicorn Segment_analysis_server_openAI:app --reload