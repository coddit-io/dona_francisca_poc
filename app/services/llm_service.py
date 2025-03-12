"""
LLM service for the Doña Francisca Ship Management System POC.

This module provides a unified interface for interacting with LLMs through LangChain,
supporting both conversational QA and structured data extraction.
"""

import json
import logging
import os
import time
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

# Model configuration
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "gemini")

# Set default providers for specific tasks
CHAT_PROVIDER = "gemini"  # Use Gemini for chat interface
EXTRACTION_PROVIDER = "openai"  # Use OpenAI for data extraction

# Type variable for Pydantic model types
ModelType = TypeVar('ModelType', bound=BaseModel)


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
        # Set default model name based on provider
        if model_name:
            self.model_name = model_name
        else:
            if provider == "openai":
                self.model_name = DEFAULT_OPENAI_MODEL
            elif provider == "gemini":
                self.model_name = DEFAULT_GEMINI_MODEL
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        # Initialize the LLM
        self.llm = self._get_llm()

    def _get_llm(self):
        """
        Initialize the appropriate LLM based on provider and model preferences.

        Returns:
            BaseChatModel: Initialized LLM instance
        """
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
            
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0.3,
                api_key=api_key
            )

        elif self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in environment variables")
            
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
                                max_retries: int = 2) -> Union[ModelType, Dict[str, Any]]:
        """
        Extract structured data from content using the LLM and a Pydantic model.

        Args:
            content (str): The content to extract data from
            output_model (Type[ModelType]): Pydantic model class defining the output structure
            instructions (Optional[str]): Additional instructions for extraction
            max_retries (int): Maximum number of retries for API errors

        Returns:
            Union[ModelType, Dict[str, Any]]: Extracted data as Pydantic model or error dict
        """
        retry_count = 0
        
        # Handle large content by taking portions 
        if len(content) > 15000:
            logger.info(f"Large content ({len(content)} chars). Using summarization.")
            # Take first and last parts of the content
            third = len(content) // 3
            content = content[:third] + "\n\n[...Content trimmed...]\n\n" + content[-third:]

        while retry_count <= max_retries:
            try:
                # Create a Pydantic output parser
                parser = PydanticOutputParser(pydantic_object=output_model)

                # Get model field descriptions for the prompt
                model_schema = self._get_model_schema_description(output_model)

                # Create basic instructions
                base_instructions = f"""
                Extract the following information from the provided content.
                {model_schema}
                
                Format your response as a valid JSON object with these exact field names.
                If information is not found, use null for that field.
                """

                # Combine with any additional instructions
                full_instructions = f"{base_instructions}\n\n{instructions}" if instructions else base_instructions

                # Create the prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Extract structured information from the provided document.
                    Identify key information matching the requested fields.
                    Return your response as a valid JSON object."""),
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

                # Parse the result into the Pydantic model
                parsed_data = output_model(**result)
                
                return parsed_data

            except ValidationError as ve:
                logger.error(f"Validation error: {str(ve)}")
                return {"error": f"Validation error: {str(ve)}"}
                
            except Exception as e:
                error_str = str(e)
                retry_count += 1
                
                # If max retries exceeded
                if retry_count > max_retries:
                    logger.error(f"Maximum retries ({max_retries}) exceeded: {error_str}")
                    return {"error": f"Failed to extract structured data: {error_str}"}
                
                # Simple backoff and retry
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"Error, retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
                
                # If content is still long, reduce more on retry
                if len(content) > 10000:
                    quarter = len(content) // 4
                    content = content[:quarter] + "\n\n[...]\n\n" + content[-quarter:]
                    logger.info(f"Reducing content size to {len(content)} characters for retry")

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
        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Remove markdown code block markers and try again
        try:
            cleaned_text = text.replace("```json", "").replace("```", "").strip()
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
