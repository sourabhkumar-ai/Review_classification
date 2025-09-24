# agent_client.py
import asyncio
from Cummulative_analysis_class_str import Segment_Analysis

if __name__ == "__main__":
    review1 = {"text":"The service was excellent and the staff were very friendly.","sentiment": "Positive", "tags": "Customer Support"}

    review2 = {"text":"The food was terrible and it took forever to arrive.","sentiment": "Negative", "tags": "Other"}

    review3 = {"text":"As someone with a chronic illness, remembering to take my meds and track my symptoms can be overwhelming. This app has been a lifesaver. The reminders are reliable, and logging my data is simple. It's great to have all my health info in one place, and my doctor loves that she can see my progress in real time. It is given me so much peace of mind.","sentiment": "Positive", "tags": "Features"}

    review4 = {"text":"The meditation sessions are excellent, and the sleep tracking is pretty accurate. However, the app feels a bit dated. Navigating between different sections isn't as smooth as it should be, and sometimes my data from my wearable doesn't sync properly, which is frustrating. It gets the job done, but it could be much better.","sentiment": "Neutral", "tags": "UI/UX"}

    review5 = {"text":"I downloaded this app to track my calories, but quickly uninstalled it. The free version bombards you with pop-up ads every two minutes. Worst of all, the privacy policy is incredibly vague, and I was uncomfortable with the amount of personal data it was collecting. I felt like my health information wasn't safe. A truly terrible experience","sentiment": "Negative", "tags": "Privacy/Security"}
    
    review6 = {"text":"UI-UX is good and mobile app is very slow","sentiment":"Neutral","tags":["UI/UX","Mobile"]}

    rev_list = [review1,review2,review3,review4,review5,review6]

    Analysis_json = {
                        "total_reviews": 1000,
                        "sentiment_distribution": {
                            "Positive": 600,
                            "Negative": 300,
                            "Neutral": 100
                        },
                        "category_distribution": {
                            "UI/UX": {"Positive": 200, "Negative": 50, "Neutral": 20},
                            "Performance": {"Positive": 150, "Negative": 100, "Neutral": 30},
                        }
                    }

    segment_analysis_obj = Segment_Analysis()
    chat_prompt = segment_analysis_obj.Prompt_Generator()

    # for review in rev_list:
    
        # payload_dict = {"text": review}
        # payload = ReviewRequest(**payload_dict)
    result = segment_analysis_obj.Cummulative_analysis_Generator(rev_list,chat_prompt,Analysis_json)

    print(result)


#However the name of the fields for segment should be tags and 
#their should be comma separated or array of string mentioning highlights on the summary in terms of tags. 
# DO you mean to provide the context on basis of tags from the summary or review? 
    
