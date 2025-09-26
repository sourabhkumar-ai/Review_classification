# llm_server.py
import os
import traceback
from dotenv import load_dotenv

from fastapi import FastAPI

from pydantic import BaseModel, ValidationError
from pydantic import BaseModel, Field

from typing import Annotated, Literal,List,Dict

from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
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
AppComponent = Literal["UI/UX", "Performance", "Bugs", "Features", "Privacy/Security", "Customer Support","Checkout","Feature_request","Mobile","Search"]
SentimentComponent = Literal["Positive","Negative","Neutral"]

# Define the Pydantic model for the incoming reviews and the output of reviews which is helpful in getting a structured data.
class ReviewRequest(BaseModel):
    text: str

class Reviews(BaseModel):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[SentimentComponent, "The overall sentiment of the review (positive, negative, or neutral)"]
    tags: Annotated[AppComponent, "The specific part of the app the review is primarily about (e.g., UI/UX, Performance, Bugs, Features, Privacy/Security, Customer Support,Checkout ,Feature_request,Mobile,Search)"]
    # highlights : Annotated[str | List[str],"Highlight the context from the summary on basis of tags, It can be a string, phrase or sentence "]

class ComprehensiveSummary(BaseModel):
    total_reviews: int
    sentiment_distribution: Dict[SentimentComponent, int]
    category_distribution: Dict[str, Dict[AppComponent, int]]
    summary: str
    key_insights: List[str]

class Segment_Analysis:
    
    def __init__(self):
        self.request = None
        self.chat_prompt = None
        # Model initialization : 
        try:
            self.llm = init_chat_model("gpt-4", 
                                model_provider="openai",
                                temperature=0.7,
                                max_tokens=512,
                                top_p=0.95,
                                )
        except ValueError as e:
            print(f"ValueError during model init: {e}")
            self.llm = None
            error_message = "Invalid parameters for model initialization."

        except ConnectionError as e:
            print(f"ConnectionError: {e}")
            self.llm = None
            error_message = "Failed to connect to model provider."

        if self.llm is None:
            raise RuntimeError(f"Model initialization failed: {error_message}")

    def Prompt_Generator(self):
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You are an assistant analyzing app reviews. 
                Reviews are already labeled with Sentiment (Positive/Negative/Neutral) and a Tags (UI/UX, Performance, Bugs, Features, Privacy/Security, Customer Support, Other).

                Your task:
                1. Calculate:
                   - Total number of reviews.
                   - Sentiment distribution across all reviews.
                2. Provide a comprehensive analysis with two headings:
                   - "summary": A holistic overview of the feedback.
                   - "key_insights": Bullet-pointed insights highlighting patterns and important observations.

                IMPORTANT:
                    - Output must be **valid JSON only**.
                    - Do not include explanations, markdown, code fences, or text outside the JSON object.
                    - The JSON object must strictly match this schema:
                {{
                  "total_reviews": <int>,
                  "sentiment_distribution": 
                  {{
                      "Positive": <int>,
                      "Negative": <int>,
                      "Neutral": <int>
                  }},
                  
                  "summary": "<overall summary>",
                  "key_insights": ["point1", "point2"]
                }}
                """
            ),
            HumanMessagePromptTemplate.from_template("{reviews}")
        ])
    def Cumm_Prompt_Generator(self):
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """
                You are an assistant analyzing app reviews. 
                Chunk based reviews summary is provided .

                Your task:
                1. Provide a comprehensive analysis with two headings:
                   - "summary": A holistic overview of the feedback.
                   - "key_insights": Bullet-pointed insights highlighting patterns and important observations.

                IMPORTANT:
                    - Output must be **valid JSON only**.
                    - Do not include explanations, markdown, code fences, or text outside the JSON object.
                    - The JSON object must strictly match this schema:
                {{
                  "Final_summary": "<overall summary>",
                  "Final_key_insights": ["point1", "point2"]
                }}
                """
            ),
            HumanMessagePromptTemplate.from_template("{summaries}")
        ])
    def Sentiment_and_Segment_Generator(self, request, chat_prompt):
        parser = JsonOutputParser(pydantic_object=Reviews)

        # Chain building.
        llm = self.llm 
        try:
            if not hasattr(request, 'text') or not isinstance(request.text, str):
                raise ValueError("Invalid or missing 'text' in request")
            chain = chat_prompt | llm | parser
            raw_response = chain.invoke({"text": request.text})
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
            
    def Cummulative_analysis_Generator_parallel(self, chunks,chat_prompt,Cummaltive_chain_prompt):
        parser = JsonOutputParser(pydantic_object=ComprehensiveSummary)
        Cummulative_summary = []
        results,final_summary = {},{}
        # Chain building.
        llm = self.llm 
        chain = chat_prompt | llm | parser
        Cummaltive_summary_chain = Cummaltive_chain_prompt | llm | parser
        if not chunks:
            return {}, {"summary": "", "key_insights": []} 
        try:
            
            if isinstance(chunks, list):
                results=[]
                parallel_runner = RunnableParallel({
                    f"chunk_{i+1}": RunnableLambda(lambda _: {"reviews": chunk}) | chain
                    for i, chunk in enumerate(chunks, start=1)
                })
                results = parallel_runner.invoke({})
                for res in results.values():
                    Cummulative_summary.append(res["summary"])
                
                final_summary = Cummaltive_summary_chain.invoke({
                    "summaries": "\n".join(Cummulative_summary)
                })
                
                return results,final_summary

        
        except AttributeError as e:
            print(f"AttributeError: {e}")
            return {"error": "Invalid request: missing 'text'"}
        
    def Cummulative_analysis_Generator_sequential(self, request,chat_prompt):
        parser = JsonOutputParser(pydantic_object=ComprehensiveSummary)
        
        # Chain building.
        llm = self.llm 
        chain = chat_prompt | llm | parser
        try:
            #If Single review passed
            if isinstance(request, str):
                raw_response = chain.invoke(request)
                return raw_response
            
            # If list of reviews are passed
            if isinstance(request, list):
                results=[]                
                results = chain.batch(request)
                return results

        
        except AttributeError as e:
            print(f"AttributeError: {e}")
            return {"error": "Invalid request: missing 'text'"}