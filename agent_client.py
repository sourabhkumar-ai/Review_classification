# agent_client.py

import requests
import json


def analyze_review_with_api(review_text: str):
    """Sends a review to the local server for analysis."""
    url = "http://127.0.0.1:8000/analyze_review"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"text": review_text})

    try:
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}


if __name__ == "__main__":
    review1 = "The service was excellent and the staff were very friendly."

    review2 = "The food was terrible and it took forever to arrive."

    review3 = "As someone with a chronic illness, remembering to take my meds and track my symptoms can be overwhelming. This app has been a lifesaver. The reminders are reliable, and logging my data is simple. It's great to have all my health info in one place, and my doctor loves that she can see my progress in real time. It is given me so much peace of mind."

    review4 = "The meditation sessions are excellent, and the sleep tracking is pretty accurate. However, the app feels a bit dated. Navigating between different sections isn't as smooth as it should be, and sometimes my data from my wearable doesn't sync properly, which is frustrating. It gets the job done, but it could be much better."

    review5 = "I downloaded this app to track my calories, but quickly uninstalled it. The free version bombards you with pop-up ads every two minutes. Worst of all, the privacy policy is incredibly vague, and I was uncomfortable with the amount of personal data it was collecting. I felt like my health information wasn't safe. A truly terrible experience"

    print("Analyzing review 1")
    result1 = analyze_review_with_api(review1)
    print(result1)

    print("\nAnalyzing review 2")
    result2 = analyze_review_with_api(review2)
    print(result2)

    print("Analyzing review 3")
    result3 = analyze_review_with_api(review3)
    print(result3)

    print("Analyzing review 4")
    result4 = analyze_review_with_api(review4)
    print(result4)

    print("Analyzing review 5")
    result5 = analyze_review_with_api(review5)
    print(result5)
