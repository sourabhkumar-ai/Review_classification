# agent_client.py
import re
import time
import json
from Cummulative_analysis_class_str import Segment_Analysis

if __name__ == "__main__":
    review1 = {"text":"The service was excellent and the staff were very friendly.","sentiment": "Positive", "tags": "Customer Support"}

    review2 = {"text":"The food was terrible and it took forever to arrive.","sentiment": "Negative", "tags": "Other"}

    review3 = {"text":"As someone with a chronic illness, remembering to take my meds and track my symptoms can be overwhelming. This app has been a lifesaver. The reminders are reliable, and logging my data is simple. It's great to have all my health info in one place, and my doctor loves that she can see my progress in real time. It is given me so much peace of mind.","sentiment": "Positive", "tags": "Features"}

    review4 = {"text":"The meditation sessions are excellent, and the sleep tracking is pretty accurate. However, the app feels a bit dated. Navigating between different sections isn't as smooth as it should be, and sometimes my data from my wearable doesn't sync properly, which is frustrating. It gets the job done, but it could be much better.","sentiment": "Neutral", "tags": "UI/UX"}

    review5 = {"text":"I downloaded this app to track my calories, but quickly uninstalled it. The free version bombards you with pop-up ads every two minutes. Worst of all, the privacy policy is incredibly vague, and I was uncomfortable with the amount of personal data it was collecting. I felt like my health information wasn't safe. A truly terrible experience","sentiment": "Negative", "tags": "Privacy/Security"}
    
    review6 = {"text":"UI-UX is good and mobile app is very slow","sentiment":"Neutral","tags":["UI/UX","Mobile"]}

    # rev_list = [review1,review2,review3,review4,review5,review6]
    rev_list = []

    with open("reviews_dataset.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue

            # Extract JSON part with regex (everything after = and starting with {)
            match = re.search(r"(\{.*\})", line)
            if match:
                dict_str = match.group(1)
                dict_str = dict_str.replace("'", '"')
                try:
                    review_dict = json.loads(dict_str)
                    rev_list.append(review_dict)
                except json.JSONDecodeError as e:
                    print(f"Skipping bad JSON line ({e})")#{line[:50]}
            

    # print(rev_list)

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

    def chunk_list_dynamic(data, num_chunks=10):
        """Split data into `num_chunks` nearly equal parts."""
        chunk_size = max(1, len(data) // num_chunks)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    chunks = list(chunk_list_dynamic(rev_list, num_chunks=2))
    result_list = []
    counter = 0

    #Below Method is to process chunks Parallely:
    start = time.perf_counter()
    result = segment_analysis_obj.Cummulative_analysis_Generator_parallel(chunks,chat_prompt)
    with open("analysis_output_parallel_request_2.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=4, ensure_ascii=False))
    end = time.perf_counter()
    time_consumed = end-start
    print(f"Time Consumed = {time_consumed}")



    # Below method is to process chunks sequentially:
    start = time.perf_counter()
    for chunk in chunks:
        print(f"Analysing..........................................................................{counter}")
        result = segment_analysis_obj.Cummulative_analysis_Generator_sequential(chunks,chat_prompt)
        result_list.append(result)
        counter+=1
        with open("analysis_output_chunk_by_chunk.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(result, indent=4, ensure_ascii=False))
    end = time.perf_counter()
    time_consumed = end-start
    print(f"Time Consumed = {time_consumed}")


'''
To be done:
chunk number should be decided dynamically
pick each summary for every chunk and process it for a cummulative summary of reach chunk.

'''

    
