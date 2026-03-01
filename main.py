from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

client = OpenAI(api_key=os.getenv("AI_API_TOKEN"))

app = FastAPI()

# Request model
class CommentRequest(BaseModel):
    comment: str

@app.post("/comment")
async def analyze_sentiment(data: CommentRequest):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze sentiment of this comment and respond in JSON with fields sentiment (positive/negative/neutral) and rating (1-5): {data.comment}",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"]
                    }
                }
            }
        )

        return JSONResponse(content=response.output_parsed)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))