from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tweepy
import torch
from typing import List
import os
from dotenv import load_dotenv
from targets import VALID_TARGETS 
import re
from typing import Optional
from time import sleep
from tweepy.errors import TooManyRequests
import time
from datetime import datetime, timedelta

load_dotenv()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables to track rate limiting
tweet_request_count = 0
tweet_last_request_time = None
TWEET_RATE_LIMIT_WINDOW = 900  # 15 minutes
TWEET_MAX_REQUESTS_PER_WINDOW = 180


# Add Twitter API credentials setup
def get_twitter_client():
    return tweepy.Client(
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
        wait_on_rate_limit=True  # This tells tweepy to automatically wait when rate limited
    )

class TweetInput(BaseModel):
    tweet_url: str
    target: Optional[str] = None

def check_tweet_rate_limit():
    global tweet_request_count, tweet_last_request_time
    
    current_time = datetime.now()
    
    # Reset counter if we're in a new window
    if tweet_last_request_time is None or current_time - tweet_last_request_time > timedelta(seconds=TWEET_RATE_LIMIT_WINDOW):
        tweet_last_request_time = current_time
        tweet_request_count = 0
        return True
    
    # Check if we've exceeded our rate limit
    if tweet_request_count >= TWEET_MAX_REQUESTS_PER_WINDOW:
        time_since_first = (current_time - tweet_last_request_time).total_seconds()
        if time_since_first < TWEET_RATE_LIMIT_WINDOW:
            return False
        else:
            # Reset for new window
            tweet_last_request_time = current_time
            tweet_request_count = 0
    
    return True


def extract_tweet_id(tweet_url: str) -> str:
    """Extract tweet ID from various Twitter URL formats."""
    patterns = [
        r'twitter\.com/\w+/status/(\d+)',
        r'x\.com/\w+/status/(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, tweet_url)
        if match:
            return match.group(1)
    raise ValueError("Invalid Tweet URL format")

    

@app.post("/analyze_tweet/")
async def analyze_tweet(input_data: TweetInput):
    global tweet_request_count
    
    if not check_tweet_rate_limit():
        raise HTTPException(
            status_code=429,
            detail=f"Tweet analysis rate limit reached. Please wait {TWEET_RATE_LIMIT_WINDOW - (datetime.now() - tweet_last_request_time).total_seconds():.0f} seconds."
        )
    
    try:
        tweet_id = extract_tweet_id(input_data.tweet_url)
        client = get_twitter_client()
        
        try:
            tweet = client.get_tweet(tweet_id, tweet_fields=['text'])
            tweet_request_count += 1
            
            if not tweet or not tweet.data:
                raise HTTPException(status_code=404, detail="Tweet not found")
            
            tweet_text = tweet.data.text
            
            stance_input = StanceInput(
                statement=tweet_text,
                target=input_data.target
            )
            
            # Reuse the predict_stance function
            result = await predict_stance(stance_input)
            result["tweet_text"] = tweet_text
            
            return result
            
        except tweepy.TooManyRequests:
            raise HTTPException(
                status_code=429,
                detail="Twitter API rate limit reached. Please try again in a few minutes."
            )
            
        except tweepy.TwitterServerError:
            raise HTTPException(
                status_code=503,
                detail="Twitter API is currently unavailable. Please try again later."
            )
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing tweet: {str(e)}")

# -----------  Stance Detection Endpoints -----------

# Load the tokenizer and model
model_name = "./models/final_stance_model" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)  

class StanceInput(BaseModel):
    statement: str
    target: str = None  # Make target optional


def get_stance_relevant_attention(attentions, logits, tokens):
    # Get the last layer's attention patterns
    last_layer = attentions[-1]  # Shape: [batch, num_heads, seq_len, seq_len]
    
    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Average across heads but weight them based on their contribution to the predicted class
    class_attention = torch.zeros(last_layer.size(-1))
    
    for head_idx in range(last_layer.size(1)):
        head_attention = last_layer[0, head_idx]  # Get attention weights for this head
        # Weight the attention based on how much this head contributes to the predicted class
        head_contribution = head_attention.sum(dim=-1)
        class_attention += head_contribution
    
    # Convert to numpy and normalize
    attention_weights = class_attention.detach().cpu().numpy()
    
    # Normalize to preserve relative importance
    attention_weights = attention_weights / attention_weights.sum()
    
    return attention_weights

@app.get("/targets")
async def get_targets():
    return {"targets": VALID_TARGETS}

stance_mapping = {0: "FAVOR", 1: "AGAINST", 2: "NONE"}


@app.post("/predict/")
async def predict_stance(input_data: StanceInput):
    try:
        if not input_data.statement.strip():
            raise HTTPException(status_code=400, detail="Input statement is empty.")

        # Checks if the target is in the predefined list or is a custom target
        if input_data.target not in VALID_TARGETS:
            detected_target = input_data.target
        else:
            detected_target = input_data.target

        # Combine target and statement as in training
        combined_input = f"{detected_target} </s> {input_data.statement}"

        # Tokenize the combined input
        inputs = tokenizer(
            combined_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        # Move model and inputs to the appropriate device (if the device has the resources)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Perform inference
        outputs = model(**inputs)
        logits = outputs.logits
        attentions = outputs.attentions

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        prediction = torch.argmax(logits, dim=1).item()

        detected_stance = stance_mapping.get(prediction, "Unknown")

        if attentions is not None:
            # Get stance-specific attention weights
            attention_weights = get_stance_relevant_attention(attentions, logits, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
            
            # Reduce weights for special tokens while preserving relative weights for content tokens
            special_tokens = {'[CLS]', '[SEP]', '</s>', '<s>', '/', 's'}
            token_list = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            for idx, token in enumerate(token_list):
                if token in special_tokens:
                    attention_weights[idx] *= 0.1  # Reduce special token importance
                    
            # Re-normalize after adjusting special tokens
            attention_weights = attention_weights / attention_weights.sum()
            
            return {
                "stance": detected_stance,
                "target": detected_target,
                "probabilities": {
                    "Support": probabilities[0],
                    "Against": probabilities[1],
                    "Neutral": probabilities[2],
                },
                "attention": attention_weights.tolist(),
                "tokens": token_list,
            }

    except Exception as e:
        print(f"Error in predict_stance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

