import logging
import os
import sys
import boto3
from dotenv import load_dotenv
import watchtower
from pydantic import BaseModel
from typing import Literal
from pydantic_models import TextEvent

# 1. Load environment variables dynamically based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTENV_PATH = os.path.join(SCRIPT_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

# 2. Initialize the CloudWatch Logs client
TARGET_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
cw_client = boto3.client("logs", region_name=TARGET_REGION)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger that streams to both stdout and AWS CloudWatch."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicating handlers if the logger was already initialized
    if not logger.handlers:

        # --- Handler 1: Local Console (stdout) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] - %(message)s"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # --- Handler 2: AWS CloudWatch Logs ---
        cw_handler = watchtower.CloudWatchLogHandler(
            boto3_client=cw_client,
            log_group_name="rag-application-logs",
            stream_name="app-instance-stdout",
            use_queues=True,  # Keeps log shipping asynchronous for optimal performance
        )
        cw_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        cw_handler.setFormatter(cw_format)
        logger.addHandler(cw_handler)

    return logger

# 3. Generic Logging Function for any Pydantic Model
def log(
    prefix_message: str = "",
    level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    model: BaseModel = TextEvent(),
) -> None:
    """
    Extracts data from any Pydantic model, formats it as JSON, 
    and sends it to the CloudWatch/Console logger named after the model.
    """
    # Safeguard check to ensure a valid Pydantic instance is passed
    if not isinstance(model, BaseModel):
        raise TypeError(f"Expected a Pydantic BaseModel instance, got {type(model)}")

    # Extract the model's class name dynamically for clean log categorization
    model_name = model.__class__.__name__
    logger = get_logger(model_name)

    # Handle serialization smoothly for both Pydantic v2 (model_dump_json) and v1 (json)
    if hasattr(model, "model_dump_json"):
        model_json = model.model_dump_json()
    else:
        model_json = model.json()  # Fallback for Pydantic v1

    # Format the final log payload
    log_message = f"{prefix_message} | {model_json}" if prefix_message else model_json

    # Dynamically locate and execute the requested logging level
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(log_message)


