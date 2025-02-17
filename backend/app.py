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


# Initialize FastAPI app
app = FastAPI()


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# New variables to track rate limiting
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
            
            # Use the existing stance detection logic
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

# ----------- Code Block 1: Stance Detection Endpoints -----------

# Load the tokenizer and model
model_name = "./models/final_stance_model"  # Path to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)  

class StanceInput(BaseModel):
    statement: str
    target: str = None  # Make target optional


@app.get("/targets")
async def get_targets():
    return {"targets": VALID_TARGETS}

stance_mapping = {0: "FAVOR", 1: "AGAINST", 2: "NONE"}


@app.post("/predict/")
async def predict_stance(input_data: StanceInput):
    try:
        if not input_data.statement.strip():
            raise HTTPException(status_code=400, detail="Input statement is empty.")

        # Check if the target is in the predefined list or is a custom target
        if input_data.target not in VALID_TARGETS:
            # Custom target is allowed, no need to check against list
            detected_target = input_data.target
        else:
            # Valid predefined target
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

        # Move model and inputs to the appropriate device
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

        # Process attention weights
        if attentions is not None:
            last_attention = attentions[-1]
            attention_scores = last_attention[0][0]
            attention_scores = attention_scores.detach().cpu().numpy()
            word_attention = attention_scores.mean(axis=0)
            word_attention = word_attention.flatten()[:len(inputs['input_ids'][0]) - 2]
        else:
            word_attention = []

        return {
            "stance": detected_stance,
            "target": detected_target,
            "probabilities": {
                "Support": probabilities[0],
                "Against": probabilities[1],
                "Neutral": probabilities[2],
            },
            "attention": word_attention.tolist(),
            "tokens": tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist()),
        }

    except Exception as e:
        print(f"Error in predict_stance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

