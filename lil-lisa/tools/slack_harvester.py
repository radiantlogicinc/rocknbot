"""
Utility to harvest complete conversation history from Slack channels
"""

import os
import sys
import argparse
from argparse import Namespace
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

from src import utils

class SlackHarvester:
    def __init__(self, token: str = None):
        """Initialize the Slack harvester with a bot token"""
        self.token = token or os.environ.get('SLACK_BOT_TOKEN')
        if not self.token:
            raise ValueError("Slack bot token is required. Set SLACK_BOT_TOKEN environment variable or pass token to constructor.")
        self.client = WebClient(token=self.token)

        # This handler does retries when HTTP status 429 is returned
        rate_limit_handler = RateLimitErrorRetryHandler(max_retry_count=1)
        # Enable rate limited error retries as well
        self.client.retry_handlers.append(rate_limit_handler)

    def get_channel_history(self, channel_id: str, oldest: float = 0) -> List[Dict[Any, Any]]:
        """
        Fetch complete conversation history from a Slack channel
        
        Args:
            channel_id (str): The ID of the channel to fetch history from
            oldest (float): Unix timestamp to start fetching from (0 means from the beginning)
            
        Returns:
            List[Dict]: List of message objects with their replies
        """
        messages = []
        cursor = None
        
        while True:
            try:
                response = self.client.conversations_history(
                    channel=channel_id,
                    cursor=cursor,
                    oldest=oldest,
                    limit=1000  # Max messages per request
                )
                
                # Process each message and fetch its replies if it's a parent message
                for msg in response['messages']:
                    if 'thread_ts' in msg and msg['thread_ts'] == msg['ts']:  # This is a parent message
                        try:
                            # Fetch the entire thread
                            thread = self.client.conversations_replies(
                                channel=channel_id,
                                ts=msg['ts']
                            )
                            msg['replies'] = thread['messages'][1:]  # Skip the parent message
                        except SlackApiError as e:
                            utils.logger.error(f"Error fetching thread replies: {str(e)}")
                    messages.append(msg)
                
                # Check if there are more messages to fetch
                if not response['has_more']:
                    break
                    
                # Get cursor for next batch
                cursor = response['response_metadata']['next_cursor']
                
                # Respect rate limits
                time.sleep(1)
                
            except SlackApiError as e:
                utils.logger.error(f"Error fetching Slack history: {str(e)}")
                raise
        
        return messages

    def save_messages_to_file(self, messages: List[Dict], output_file: str):
        """
        Save harvested messages to a file with proper threading format
        
        Args:
            messages (List[Dict]): List of message objects
            output_file (str): Path to output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for msg in messages:
                    # Skip messages that are replies (they'll be handled with their parent)
                    if 'thread_ts' in msg and msg['thread_ts'] != msg['ts']:
                        continue
                        
                    # Write the main message
                    timestamp = datetime.fromtimestamp(float(msg['ts'])).strftime('%Y-%m-%d %H:%M:%S')
                    text = msg.get('text', '')
                    user = msg.get('user', 'UNKNOWN')
                    f.write(f"\n[{timestamp}] {user}: {text}\n")
                    
                    # Write replies if they exist
                    if 'replies' in msg:
                        for reply in msg['replies']:
                            reply_timestamp = datetime.fromtimestamp(float(reply['ts'])).strftime('%Y-%m-%d %H:%M:%S')
                            reply_text = reply.get('text', '')
                            reply_user = reply.get('user', 'UNKNOWN')
                            f.write(f"    ↳ [{reply_timestamp}] {reply_user}: {reply_text}\n")
                        f.write("\n")  # Add extra line break after thread
                    
        except IOError as e:
            utils.logger.error(f"Error saving messages to file: {str(e)}")
            raise

def harvest_slack_messages(args, oldest_timestamp):
    harvester = SlackHarvester(token=args.slack_token)
    time_range = f"the past {args.days} days" if args.days else "all time"
    utils.logger.info(f"Fetching messages from channel {args.channel} for {time_range}")
    messages = harvester.get_channel_history(args.channel, oldest=oldest_timestamp)
    utils.logger.info(f"Retrieved {len(messages)} messages")

    harvester.save_messages_to_file(messages, args.output)
    utils.logger.info(f"Messages saved to {args.output}")

def how_to_use(args: Namespace):
    """
    function to demonstrate example usage of SlackHarvester
    """
    # Calculate the timestamp for N days ago if specified
    oldest_timestamp = 0  # Default to beginning of time
    if args.days is not None:
        days_ago = datetime.now() - timedelta(days=args.days)
        oldest_timestamp = days_ago.timestamp()

    try:
        harvest_slack_messages(args, oldest_timestamp)
    except Exception as e:
        utils.logger.error(f"Error in Slack harvester: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Harvest messages from a Slack channel')
    parser.add_argument('--channel', type=str, required=True, help='Slack channel ID')
    parser.add_argument('--slack-token', type=str, required=True, help='Slack Bot User OAuth Token')
    parser.add_argument('--days', type=int, help='Number of days to look back (if not specified, fetches entire history)')
    parser.add_argument('--output', type=str, default='slack_history.txt', help='Output file path (default: slack_history.txt)')

    args = parser.parse_args()
    sys.exit(how_to_use(args))
