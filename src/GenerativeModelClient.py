import time
from collections import deque

from openai import OpenAI
import google.generativeai as genai
import anthropic
   
# Define a class that in the constructor receives api_key, model, system_prompt, max_tokens, and temperature,
# and offers a generate method that receives user_prompt and returns a text and the complete response from the API
class GenerativeModelClient:
    def __init__(self, model, system_prompt, max_tokens, temperature):
        # Depending on the model, the client is initialized with the appropriate parameters
        # Specific cases are considered for different APIs, and if the string is unknown
        # it is assumed that we are working locally with a test model in llm-studio
        match model:
            case "gpt-4o" | "ft:gpt-4o-mini-2024-07-18:personal:fallacyes-political:AT6jWBUt"| "o1-mini" | "o1-preview":
                self.client = OpenAI(                
                    api_key= "??????",
                    organization="org-mE9AXPapXwAQqIhYiR0lBfuo",
                )
                self.rpm = -1 # No call limit per minute
            case "grok-beta":
                self.client = OpenAI(                
                    api_key= "??????",
                    base_url="https://api.x.ai/v1"
                )
                self.rpm = 60 # Call limit of 60 per minute
            case "gemini-1.5-pro-002":     
                genai.configure(api_key="??????",)           
                self.client = genai.GenerativeModel(model_name="gemini-1.5-pro-002",
                    system_instruction=system_prompt,
                    generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        ))
                self.rpm = -1
            case "gemini-exp-1114":     
                genai.configure(api_key= "??????",)           
                self.client = genai.GenerativeModel(model_name="gemini-exp-1114",
                    system_instruction=system_prompt,
                    generation_config=genai.types.GenerationConfig(
                            max_output_tokens=max_tokens,
                            temperature=temperature,
                        ))
                self.rpm = 4
            case "anthropic/claude-3.5-sonnet":
                self.client = anthropic.Anthropic(              
                    api_key= "??????"
                )
                self.rpm = -1
            case "meta/llama-3.1-405b-instruct":
                self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", 
                                     api_key= "??????",)                                     
                self.rpm = -1                
            case "nvidia/llama-3.1-nemotron-70b-instruct" | "meta/llama-3.1-405b-instruct":
                self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",                  
                                     api_key= "??????",)
                self.rpm = -1                
            case _:
                self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                self.rpm = -1

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.call_times = deque()

    def generate(self, user_prompt, progress_bar=None):
        current_time = time.time()
        if self.rpm > 0:
            first = True
            while len(self.call_times) >= self.rpm and current_time - self.call_times[0] < 60:
                if first:
                    if progress_bar:
                        progress_bar.set_postfix_str("Waiting to comply with the call limit per minute...")
                    else:
                        print("Waiting to comply with the call limit per minute...")
                    first = False
                time.sleep(1)
                current_time = time.time()
            if len(self.call_times) >= self.rpm:
                self.call_times.popleft()
            self.call_times.append(current_time)
            if progress_bar:
                progress_bar.set_postfix_str("")

        match self.model:
            case "gemini-1.5-pro-002" | "gemini-exp-1114":
                response = self.client.generate_content(user_prompt)
                return response.text, response       
                 
            case "o1-mini" | "o1-preview":                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        #{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.system_prompt + "\n\n" + user_prompt}
                    ],
                    #max_tokens=self.max_tokens, # This model does not support max_tokens or temperature parameter
                    #temperature=self.temperature
                ) 
                if getattr(completion, 'error', None) and  completion.error and 'message' in completion.error and "moderation" in completion.error['message']:
                    return "MODERATED", completion
                
                return completion.choices[0].message.content, completion
            
            case "anthropic/claude-3.5-sonnet":
                completion = self.client.messages.create(
                    model = "claude-3-5-sonnet-20241022",
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=self.system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                }
                            ]
                        }
                    ]
                )
                return completion.content[0].text, completion
            
            case _:  # Here we consider the models that fit the OpenAI API
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return completion.choices[0].message.content, completion
    
    def __str__(self):
        return str(self.client)


