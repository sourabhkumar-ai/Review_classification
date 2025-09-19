# Review_classification
If want to RUN on Local
  LLM in this project is running on Local.
  
  In order to run the local LLM model needs to be downloaded on Local.
  
  Model name - TheBloke--Mistral-7B-Instruct-v0.2 
  LINK - https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
  
USing OPENAI Key :
  clone branch First_phase_v2.

Post downloding:

Open file llm_server_chain_method.py file and change the model_path.

and run this using uvicorn server.

uvicorn llm_server_chain_method:app --reload

After the successfull run;

Sample REviews are already mentioned in the agent_client.py file to test.
Else; Use Fastapi/docs to call that api.

Reponse format : 

Analyzing review 5...
{'summary': 'The review describes a negative experience with the app. The user found the free version to be bombarded by pop-up ads every two minutes. Additionally, the privacy policy was considered vague and collecting an uncomfortable amount of personal data.', 'sentiment': 'Negative', 'segment': 'Privacy/Security'}
