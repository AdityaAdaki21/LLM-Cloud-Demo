import gradio as gr
import requests
import os
import json # Import json for better error message handling

# --- Configuration ---
# Using the specific Nebius/HF router URL from your snippet
API_URL = "https://router.huggingface.co/nebius/v1/chat/completions"
# Using the model from your snippet
MODEL_ID = "google/gemma-3-27b-it-fast"
# Get Hugging Face token from environment variable/secrets
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable or secret.")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

# --- No Local Model Loading Needed ---
print(f"Application configured to use Hugging Face Inference API.")
print(f"Target Model: AgriAssist_LLM")
print(f"API Endpoint: {API_URL}")

# --- Inference Function (Using Hugging Face API) ---
def generate_response(prompt, max_new_tokens=512): # Using max_tokens from your snippet
    print(f"Received prompt: {prompt}")
    print("Preparing payload for API...")

    # Construct the payload based on the API requirements
    # NOTE: This version assumes text-only input matching the Gradio interface.
    #       To handle image input like your snippet, the Gradio interface
    #       and payload structure would need modification.
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
                # Example for multimodal if Gradio input changes:
                # "content": [
                #    {"type": "text", "text": prompt},
                #    {"type": "image_url", "image_url": {"url": "some_image_url.jpg"}}
                # ]
            }
        ],
        "model": MODEL_ID,
        "max_tokens": max_new_tokens,
        # Optional parameters you might want to add:
        # "temperature": 0.7,
        # "top_p": 0.9,
        # "stream": False # Set to True for streaming responses if API supports it
    }

    print(f"Sending request to API for model AgriAssist_LLM...")
    try:
        # Make the POST request
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        # Raise an exception for bad status codes (like 4xx or 5xx)
        response.raise_for_status()

        # Parse the JSON response
        result = response.json()
        print("API Response Received Successfully.")

        # Extract the generated text - Structure matches your snippet's expectation
        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            api_response_content = result["choices"][0]["message"]["content"]
            print(f"API generated content: {api_response_content}")
            return api_response_content
        else:
            # Handle unexpected response structure
            print(f"Unexpected API response structure: {result}")
            return f"Error: Unexpected API response structure. Full response: {json.dumps(result)}"

    except requests.exceptions.RequestException as e:
        # Handle network errors, timeout errors, invalid responses, etc.
        error_message = f"Error calling Hugging Face API: {e}"
        # Try to get more details from the response body if it exists
        error_detail = ""
        if e.response is not None:
            try:
                error_detail = e.response.json() # Try parsing JSON error
            except json.JSONDecodeError:
                error_detail = e.response.text # Fallback to raw text
        print(f"{error_message}\nResponse details: {error_detail}")
        return f"{error_message}\nDetails: {error_detail}"

    except Exception as e:
        # Handle other potential errors during processing
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

# --- Gradio Interface ---
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, label="Enter your prompt", placeholder="Type your question or instruction here..."),
    outputs=gr.Textbox(lines=8, label=f"AgriAssist_LLM Says (via API):"), # Updated label
    title=f"Chat with AgriAssist_LLM via Inference API", # Updated title
    description=("This demo sends your text to a remote server for processing."), # Updated description
    allow_flagging="never",
    examples=[ # Examples should still be relevant
        ["Explain the concept of cloud computing in simple terms."],
        ["Write Python code to list files in a directory."],
        ["What are the main benefits of using Generative AI?"],
        ["Translate 'Cloud computing offers scalability' to German."],
    ]
)

# --- Launch the App ---
# You can add share=True if you want to create a temporary public link (use with caution)
iface.launch(server_name="0.0.0.0", server_port=7860) # Makes it accessible in Codespaces/docker