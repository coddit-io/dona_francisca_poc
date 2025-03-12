"""
LLM service for the Doña Francisca Ship Management System POC.

This module provides a unified interface for interacting with LLMs through LangChain,
supporting both conversational QA and structured data extraction.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, get_type_hints

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.globals import set_debug

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
                ("system", "You are an assistant for the ship 'Doña Francisca'. Your task is to answer questions based ONLY on the information contained in the ship manual provided. If the answer cannot be found in the manual, state that you cannot find the information in the provided documentation. The manual might be in any language, but you should be able to understand it and provide answers in the same language as the user's question."),
                ("human", "SHIP MANUAL:\n{manual_content}\n\nUSER QUESTION:\n{query}\n\nANSWER:")
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
                              force_model: Optional[str] = None) -> Union[ModelType, Dict[str, Any]]:
        """
        Extract structured data from content using the LLM and a Pydantic model.
        
        Args:
            content (str): The content to extract data from
            output_model (Type[ModelType]): Pydantic model class defining the output structure
            instructions (Optional[str]): Additional instructions for extraction
            force_model (Optional[str]): Override the model to use for extraction
            
        Returns:
            Union[ModelType, Dict[str, Any]]: Extracted data as Pydantic model or error dict
        """
        try:
            # Use a specific extraction model when provider is OpenAI
            original_llm = self.llm
            if self.provider == "openai" and (force_model or self.model_name != DEFAULT_EXTRACTION_MODEL):
                extraction_model = force_model or DEFAULT_EXTRACTION_MODEL
                logger.info(f"Using specialized extraction model: {extraction_model}")
                api_key = os.getenv("OPENAI_API_KEY")
                extraction_llm = ChatOpenAI(
                    model_name=extraction_model,
                    temperature=0.2,  # Lower temperature for more precise extraction
                    api_key=api_key
                )
                self.llm = extraction_llm
            
            # Create a Pydantic output parser
            parser = PydanticOutputParser(pydantic_object=output_model)
            
            # Get model field descriptions for the prompt
            model_schema = self._get_model_schema_description(output_model)
            
            # Create the base instructions
            base_instructions = f"""
            Extract the following information from the provided content.
            Focus on understanding document structure and identifying patterns where this information might be found.
            
            {model_schema}
            
            Format your response as a valid JSON object with these exact field names.
            If information is not found, use null for that field.
            """
            
            # Combine with any additional instructions
            full_instructions = f"{base_instructions}\n\n{instructions}" if instructions else base_instructions
            
            # Create the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an assistant specialized in extracting structured information from technical documents in any language.

Your task is to identify patterns of data presentation and information structures in documents regardless of their language.

When extracting data:
1. Analyze the general structure of the document to find relevant sections
2. Identify common patterns like "label: value" or specification lists
3. Look for sections with titles or headers that indicate information categories
4. Pay attention to the hierarchical organization of data (sections, subsections, etc.)
5. Extract values maintaining their original format, including units
6. If a requested value is not in the document, use null
7. Return your response as a valid JSON object

Focus on understanding and extracting the underlying structure of the data, working with whatever language the document is written in."""),
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
            
            # Restore original LLM if we temporarily changed it
            if self.llm is not original_llm:
                self.llm = original_llm
            
            return parsed_data
            
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return {"error": f"Validation error: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error extracting structured data: {str(e)}", exc_info=True)
            # Restore original LLM if we temporarily changed it
            if 'original_llm' in locals() and self.llm is not original_llm:
                self.llm = original_llm
            return {"error": f"Failed to extract structured data: {str(e)}"}
    
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
    service = default_llm_service
    
    # Create a new service instance if provider or model is specified
    if provider and provider != service.provider or model and model != service.model_name:
        service = LLMService(provider=provider or service.provider, model_name=model)
    
    return service.get_answer(query, manual_content) 