# agent_client.py

import requests
import json
import asyncio
from Segment_analysis_server_openAI_class_str import Segment_Analysis,ReviewRequest

if __name__ == "__main__":
    review1 = "The service was excellent and the staff were very friendly."

    review2 = "The food was terrible and it took forever to arrive."

    review3 = "As someone with a chronic illness, remembering to take my meds and track my symptoms can be overwhelming. This app has been a lifesaver. The reminders are reliable, and logging my data is simple. It's great to have all my health info in one place, and my doctor loves that she can see my progress in real time. It is given me so much peace of mind."

    review4 = "The meditation sessions are excellent, and the sleep tracking is pretty accurate. However, the app feels a bit dated. Navigating between different sections isn't as smooth as it should be, and sometimes my data from my wearable doesn't sync properly, which is frustrating. It gets the job done, but it could be much better."

    review5 = "I downloaded this app to track my calories, but quickly uninstalled it. The free version bombards you with pop-up ads every two minutes. Worst of all, the privacy policy is incredibly vague, and I was uncomfortable with the amount of personal data it was collecting. I felt like my health information wasn't safe. A truly terrible experience"
    rev_list = [review3,review4,review5]

    segment_analysis_obj = Segment_Analysis()
    chat_prompt = segment_analysis_obj.Prompt_Generator()

    for review in rev_list:
    
        payload_dict = {"text": review}
        payload = ReviewRequest(**payload_dict)
        result = asyncio.run(segment_analysis_obj.Sentiment_and_Segment_Generator(payload,chat_prompt))

        print(result)

    
