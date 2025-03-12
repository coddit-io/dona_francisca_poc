"""
LLM service for the Doña Francisca Ship Management System POC.

This module provides a unified interface for interacting with LLMs through LangChain,
supporting both conversational QA and structured data extraction.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm_service")

# Set debug mode if needed
# set_debug(True)

# Model configuration
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
DEFAULT_EXTRACTION_MODEL = "gpt-4o"  # Use GPT-4o specifically for extraction
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")

# Set default providers for specific tasks
CHAT_PROVIDER = "gemini"  # Use Gemini for chat interface
EXTRACTION_PROVIDER = "openai"  # Use OpenAI for data extraction

# Type variable for Pydantic model types
ModelType = TypeVar('ModelType', bound=BaseModel)

# Token rate limiter for managing API quota


class TokenRateLimiter:
    """
    Tracks token usage and manages backoff for rate limits.
    Implements a sliding window to track tokens per minute.
    """

    # Default token limits per minute for different providers and models
    DEFAULT_LIMITS = {
        "openai": {
            "gpt-4o": 30000,
            "gpt-4o-mini": 200000,
            "gpt-3.5-turbo": 60000,
            "default": 30000
        },
        "gemini": {
            "gemini-2.0-flash": 1000000,  # 1M TPM in free tier
            "gemini-2.0-flash-lite": 1000000,  # 1M TPM in free tier
            "gemini-2.0-pro": 1000000,  # 1M TPM in free tier for experimental version
            "gemini-1.5-flash": 1000000,  # 1M TPM in free tier
            "gemini-1.5-pro": 32000,     # 32K TPM in free tier
            "default": 32000
        }
    }

    def __init__(self):
        """Initialize the token rate limiter."""
        # Track token usage with timestamps
        self.token_usage = []
        self.request_history = []
        self.last_reset = datetime.now()

    def check_rate_limit(self, provider: str, model: str, tokens: int) -> bool:
        """
        Check if the current request would exceed rate limits.

        Args:
            provider: The LLM provider
            model: The model name
            tokens: Estimated token count for request

        Returns:
            bool: True if within rate limits, False if would exceed
        """
        now = datetime.now()
        self._prune_old_entries(now)

        # Calculate current token usage in the sliding window
        current_usage = sum(count for _, count in self.token_usage)

        # Get the token limit for this provider and model
        limit = self._get_token_limit(provider, model)

        # Check if adding these tokens would exceed the limit
        return (current_usage + tokens) <= limit

    def add_usage(self, provider: str, model: str, tokens: int):
        """
        Record token usage for rate limiting.

        Args:
            provider: The LLM provider
            model: The model name
            tokens: Token count used in request
        """
        now = datetime.now()
        self._prune_old_entries(now)

        # Add this usage to the tracking
        self.token_usage.append((now, tokens))
        self.request_history.append((now, provider, model))

    def get_backoff_time(self, provider: str, model: str, tokens: int) -> float:
        """
        Calculate the time to wait before retrying based on current usage.

        Args:
            provider: The LLM provider
            model: The model name
            tokens: Estimated token count for request

        Returns:
            float: Seconds to wait before next request to stay within limits
        """
        now = datetime.now()
        self._prune_old_entries(now)

        # Calculate current token usage
        current_usage = sum(count for _, count in self.token_usage)

        # Get the token limit for this provider and model
        limit = self._get_token_limit(provider, model)

        if current_usage + tokens <= limit:
            # No backoff needed, we're within limits
            return 0

        # Calculate how many tokens we need to wait for to expire
        excess_tokens = (current_usage + tokens) - limit

        if not self.token_usage:
            # If no usage history, use default backoff
            return 2.0

        # Sort usage by time to find the oldest entries
        sorted_usage = sorted(self.token_usage, key=lambda x: x[0])

        # Calculate how long until enough tokens expire to allow this request
        tokens_to_expire = 0
        for timestamp, count in sorted_usage:
            tokens_to_expire += count
            if tokens_to_expire >= excess_tokens:
                # This entry expiring would provide enough tokens
                wait_time = (timestamp + timedelta(minutes=1) -
                             now).total_seconds()
                # Always wait at least 1 second to be safe
                return max(1.0, wait_time)

        # If we can't determine a specific time, use an exponential backoff
        return 15.0  # Default to 15 seconds if can't calculate precisely

    def _prune_old_entries(self, now: datetime):
        """
        Remove entries older than one minute from the tracking.

        Args:
            now: Current timestamp
        """
        one_minute_ago = now - timedelta(minutes=1)

        # Keep only entries from the last minute
        self.token_usage = [(ts, count)
                            for ts, count in self.token_usage if ts >= one_minute_ago]
        self.request_history = [
            (ts, p, m) for ts, p, m in self.request_history if ts >= one_minute_ago]

    def _get_token_limit(self, provider: str, model: str) -> int:
        """
        Get the token per minute limit for a provider and model.

        Args:
            provider: The LLM provider
            model: The model name

        Returns:
            int: Token per minute limit
        """
        provider_limits = self.DEFAULT_LIMITS.get(provider.lower(), {})
        return provider_limits.get(model.lower(), provider_limits.get("default", 30000))

    def get_current_usage_info(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about current token usage.

        Args:
            provider: Optional provider to filter by

        Returns:
            Dict with usage statistics
        """
        now = datetime.now()
        self._prune_old_entries(now)

        # Get overall usage
        total_tokens = sum(count for _, count in self.token_usage)

        # Count requests by provider and model
        provider_counts = {}
        model_counts = {}

        for _, p, m in self.request_history:
            provider_counts[p] = provider_counts.get(p, 0) + 1
            model_counts[m] = model_counts.get(m, 0) + 1

        # Filter by provider if specified
        if provider:
            filtered_usage = [(ts, count) for ts, p, _ in zip(
                self.token_usage, self.request_history) if p == provider]
            provider_tokens = sum(count for _, count in filtered_usage)
        else:
            provider_tokens = total_tokens

        return {
            "total_tokens_last_minute": total_tokens,
            "provider_tokens": provider_tokens,
            "requests_by_provider": provider_counts,
            "requests_by_model": model_counts,
            "request_count": len(self.request_history)
        }


# Create a global rate limiter instance
token_rate_limiter = TokenRateLimiter()


class LLMService:
    """
    Unified service for interacting with LLMs through LangChain.
    Supports conversational QA and structured data extraction.
    """

    def __init__(self, provider: str = DEFAULT_PROVIDER, model_name: Optional[str] = None):
        """
        Initialize the LLM service.

        Args:
            provider (str): The LLM provider to use ('openai' or 'gemini')
            model_name (Optional[str]): Specific model name, or None to use default
        """
        self.provider = provider
        self.model_name = model_name or self._get_default_model()
        self.llm = self._initialize_llm()

    def _get_default_model(self) -> str:
        """
        Get the default model name based on the provider.

        Returns:
            str: Default model name
        """
        if self.provider == "openai":
            return DEFAULT_OPENAI_MODEL
        elif self.provider == "gemini":
            return DEFAULT_GEMINI_MODEL
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _initialize_llm(self):
        """
        Initialize the appropriate LLM based on provider.

        Returns:
            BaseChatModel: Initialized LLM instance
        """
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found in environment variables")

            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0.3,
                api_key=api_key
            )

        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not found in environment variables")

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.3,
                google_api_key=api_key
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_answer(self, query: str, manual_content: Dict[str, str]) -> Dict[str, Any]:
        """
        Get an answer from the LLM based on manual content.

        Args:
            query (str): The user's question
            manual_content (Dict[str, str]): Dictionary mapping manual names to their content

        Returns:
            Dict[str, Any]: Dictionary containing the answer and related metadata
        """
        if not manual_content:
            return {
                "answer": "No manual was selected. Please select a manual to search for information.",
                "sources": [],
                "error": None
            }

        try:
            # Create a single string with all manual content
            compiled_content = self._compile_manual_content(manual_content)

            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant for the ship 'Doña Francisca'. Your task is to answer questions based ONLY on the 
                 information contained in the ship manual provided. If the answer cannot be found in the manual, state that you cannot 
                 find the information in the provided documentation. The manual might be in any language, but you should be able to 
                 understand it and provide answers in the same language as the user's question."""),
                ("human",
                 "SHIP MANUAL:\n{manual_content}\n\nUSER QUESTION:\n{query}\n\nANSWER:")
            ])

            # Create the chain
            chain = prompt | self.llm | StrOutputParser()

            # Run the chain
            response = chain.invoke({
                "manual_content": compiled_content,
                "query": query
            })

            # Return the result
            return {
                "answer": response,
                "sources": list(manual_content.keys()),
                "error": None,
                "provider": self.provider,
                "model": self.model_name
            }

        except Exception as e:
            error_message = str(e)
            logger.error(f"Error calling LLM: {error_message}", exc_info=True)

            return {
                "answer": f"Sorry, I encountered an error while trying to answer your question: {error_message}.",
                "sources": [],
                "error": error_message,
                "provider": self.provider,
                "model": self.model_name
            }

    def extract_structured_data(self,
                                content: str,
                                output_model: Type[ModelType],
                                instructions: Optional[str] = None,
                                force_model: Optional[str] = None,
                                max_retries: int = 3,
                                simplified: bool = False) -> Union[ModelType, Dict[str, Any]]:
        """
        Extract structured data from content using the LLM and a Pydantic model.

        Args:
            content (str): The content to extract data from
            output_model (Type[ModelType]): Pydantic model class defining the output structure
            instructions (Optional[str]): Additional instructions for extraction
            force_model (Optional[str]): Override the model to use for extraction
            max_retries (int): Maximum number of retries for rate limit errors
            simplified (bool): Whether to use simplified prompts for lighter models

        Returns:
            Union[ModelType, Dict[str, Any]]: Extracted data as Pydantic model or error dict
        """
        retry_count = 0
        total_wait_time = 0
        max_wait_time = 120  # Maximum seconds to wait in total (2 minutes)

        # Roughly estimate the token count (1 token ≈ 4 chars)
        est_tokens = len(content) // 4

        while retry_count <= max_retries and total_wait_time < max_wait_time:
            try:
                # Use a specific extraction model when provider is OpenAI
                original_llm = self.llm
                provider = self.provider
                model_to_use = force_model or self.model_name

                if force_model:
                    logger.info(f"Using forced model: {force_model}")
                    if self.provider == "openai":
                        api_key = os.getenv("OPENAI_API_KEY")
                        extraction_llm = ChatOpenAI(
                            model_name=force_model,
                            temperature=0.2,  # Lower temperature for more precise extraction
                            api_key=api_key
                        )
                    else:
                        # For gemini or other providers
                        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
                        extraction_llm = ChatGoogleGenerativeAI(
                            model=force_model,
                            temperature=0.2,
                            google_api_key=api_key
                        )
                    self.llm = extraction_llm

                # Check rate limits before proceeding
                if not token_rate_limiter.check_rate_limit(provider, model_to_use, est_tokens):
                    # Calculate backoff time
                    backoff_time = token_rate_limiter.get_backoff_time(
                        provider, model_to_use, est_tokens)

                    # If we've already waited too long in total, fail with a clear message
                    if total_wait_time + backoff_time > max_wait_time:
                        logger.warning(
                            f"Rate limit backoff would exceed maximum wait time ({max_wait_time}s)")
                        return {"error": "Rate limits exceeded. Consider using a different model, reducing content size, or trying again later."}

                    # Wait before retrying
                    logger.info(
                        f"Rate limit approaching for {provider}/{model_to_use}. Backing off for {backoff_time:.1f}s")
                    time.sleep(backoff_time)
                    total_wait_time += backoff_time
                    continue

                # Create a Pydantic output parser
                parser = PydanticOutputParser(pydantic_object=output_model)

                # Get model field descriptions for the prompt
                model_schema = self._get_model_schema_description(output_model)

                # Create the base instructions with complexity based on the simplified flag
                if simplified:
                    # Simpler instructions for lightweight models
                    base_instructions = f"""
                    Extract the following information from the provided content.
                    {model_schema}
                    
                    Format your response as a valid JSON object with these exact field names.
                    If information is not found, use null for that field.
                    """

                    # Simpler system prompt for lightweight models
                    system_prompt = """
                    Extract structured information from the provided document.
                    Identify key information matching the requested fields.
                    Return your response as a valid JSON object.
                    """
                else:
                    # Full instructions for more powerful models
                    base_instructions = f"""
                    Extract the following information from the provided content.
                    Focus on understanding document structure and identifying patterns where this information might be found.
                    
                    {model_schema}
                    
                    Format your response as a valid JSON object with these exact field names.
                    If information is not found, use null for that field.
                    """

                    # Comprehensive system prompt for powerful models
                    system_prompt = """You are an assistant specialized in extracting structured information from technical documents in any language.

                    Your task is to identify patterns of data presentation and information structures in documents regardless of their language.

                    When extracting data:
                    1. Analyze the general structure of the document to find relevant sections
                    2. Identify common patterns like "label: value" or specification lists
                    3. Look for sections with titles or headers that indicate information categories
                    4. Pay attention to the hierarchical organization of data (sections, subsections, etc.)
                    5. Extract values maintaining their original format, including units
                    6. If a requested value is not in the document, use null
                    7. Return your response as a valid JSON object

                    Focus on understanding and extracting the underlying structure of the data, working with whatever language the document is written in."""

                # Combine with any additional instructions
                full_instructions = f"{base_instructions}\n\n{instructions}" if instructions else base_instructions

                # Adaptive content handling to prevent token limit issues
                content_size = len(content)
                if content_size > 15000 and (simplified or ("gpt-3.5" in model_to_use or "flash" in model_to_use)):
                    # For lightweight models, use a more aggressive summary approach
                    logger.info(
                        f"Large content ({content_size} chars) with lightweight model. Using aggressive summarization.")
                    # Take first quarter and last quarter of the content to stay well under limits
                    quarter = content_size // 4
                    content = content[:quarter] + \
                        "\n\n[...Content trimmed for token limits...]\n\n" + \
                        content[-quarter:]
                    # Recalculate token estimate
                    est_tokens = len(content) // 4
                elif content_size > 25000 and ("gpt-4" in model_to_use or "pro" in model_to_use):
                    # For powerful models with very large content
                    logger.info(
                        f"Very large content ({content_size} chars). Using moderate summarization.")
                    # Take first third and last third
                    third = content_size // 3
                    content = content[:third] + \
                        "\n\n[...Content trimmed for token limits...]\n\n" + \
                        content[-third:]
                    # Recalculate token estimate
                    est_tokens = len(content) // 4

                # Create the prompt with appropriate system message
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{instructions}\n\nCONTENT:\n{content}\n\nProvide only the JSON output:")
                ])

                # Create the chain with format instructions
                chain = (
                    {"instructions": lambda x: full_instructions + "\n\n" + parser.get_format_instructions(),
                     "content": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                    | self._extract_json
                )

                # Run the chain
                result = chain.invoke(content)

                # Record token usage
                token_rate_limiter.add_usage(
                    provider, model_to_use, est_tokens)

                # Parse the result into the Pydantic model
                parsed_data = output_model(**result)

                # Restore original LLM if we temporarily changed it
                if self.llm is not original_llm:
                    self.llm = original_llm

                return parsed_data

            except Exception as e:
                error_str = str(e)

                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str or "resource exhausted" in error_str.lower() or "quota" in error_str.lower():
                    retry_count += 1

                    if retry_count > max_retries:
                        logger.error(
                            f"Maximum retries ({max_retries}) exceeded for rate limit errors")
                        # Restore original LLM if needed
                        if 'original_llm' in locals() and self.llm is not original_llm:
                            self.llm = original_llm
                        return {"error": f"Failed due to rate limits after {max_retries} retries. Please try again later."}

                    # Calculate backoff time: exponential with jitter
                    import random
                    base_wait = min(30, 4 ** retry_count)  # Cap at 30 seconds
                    # Add some randomness
                    jitter = random.uniform(0.8, 1.2)
                    backoff_time = base_wait * jitter

                    logger.warning(
                        f"Rate limit hit, retrying in {backoff_time:.1f} seconds... (Attempt {retry_count}/{max_retries})")

                    # Wait before retrying
                    time.sleep(backoff_time)
                    total_wait_time += backoff_time

                    # For rate limit errors, try with a smaller chunk next time
                    if len(content) > 10000:
                        # Take first quarter and last quarter of the content
                        quarter = len(content) // 4
                        content = content[:quarter] + \
                            "\n\n[...]\n\n" + content[-quarter:]
                        logger.info(
                            f"Reducing content size to {len(content)} characters for retry")
                        # Update token estimate
                        est_tokens = len(content) // 4

                    continue

                # If not a rate limit error or max retries exceeded
                if isinstance(e, ValidationError):
                    logger.error(f"Validation error: {error_str}")
                    return {"error": f"Validation error: {error_str}"}
                else:
                    logger.error(
                        f"Error extracting structured data: {error_str}", exc_info=True)
                    # Restore original LLM if we temporarily changed it
                    if 'original_llm' in locals() and self.llm is not original_llm:
                        self.llm = original_llm
                    return {"error": f"Failed to extract structured data: {error_str}"}

    def _compile_manual_content(self, manual_content: Dict[str, str]) -> str:
        """
        Compile multiple manual contents into a single string.

        Args:
            manual_content (Dict[str, str]): Dictionary mapping manual names to their content

        Returns:
            str: Compiled content string
        """
        compiled_content = ""

        for manual_name, content in manual_content.items():
            # Add manual name as a section header
            compiled_content += f"\n\n### MANUAL: {manual_name} ###\n\n"
            compiled_content += content

        return compiled_content

    def _get_model_schema_description(self, model_class: Type[BaseModel]) -> str:
        """
        Generate a human-readable description of a Pydantic model's fields.

        Args:
            model_class (Type[BaseModel]): Pydantic model class

        Returns:
            str: Human-readable schema description
        """
        fields_info = []

        for field_name, field in model_class.model_fields.items():
            description = field.description or field_name
            fields_info.append(f"- {field_name}: {description}")

        return "\n".join(fields_info)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON from text using various fallback methods.

        Args:
            text (str): Text potentially containing JSON

        Returns:
            Dict[str, Any]: Extracted JSON data
        """
        # Log the raw text (first 500 chars)
        logger.info(f"Extracting JSON from text: {text[:500]}...")

        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Remove markdown code block markers and try again
        try:
            cleaned_text = text.replace(
                "```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # Try using regex to find JSON object
        import re
        json_pattern = re.search(r'({[\s\S]*?})(?:\s*$|\n)', text)
        if json_pattern:
            try:
                return json.loads(json_pattern.group(1))
            except json.JSONDecodeError:
                pass

        # More aggressive pattern match
        json_pattern = re.search(r'{.*}', text, re.DOTALL)
        if json_pattern:
            try:
                return json.loads(json_pattern.group(0))
            except json.JSONDecodeError:
                pass

        # If all extraction methods fail
        raise ValueError("No valid JSON found in response")


# Create a singleton instance for easy import
default_llm_service = LLMService(provider=DEFAULT_PROVIDER)


def get_answer(query: str, manual_content: Dict[str, str], provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Legacy wrapper function for backward compatibility.
    Get an answer from the LLM based on manual content.

    Args:
        query (str): The user's question
        manual_content (Dict[str, str]): Dictionary mapping manual names to their content
        provider (Optional[str]): The LLM provider to use ('openai' or 'gemini')
        model (Optional[str]): Model to use

    Returns:
        Dict[str, Any]: Dictionary containing the answer and related metadata
    """
    # Use the chat provider by default if none specified
    provider = provider or CHAT_PROVIDER
    service = LLMService(provider=provider, model_name=model)

    return service.get_answer(query, manual_content)
