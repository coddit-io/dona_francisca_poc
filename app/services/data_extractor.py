"""
Data extraction service for the Doña Francisca Ship Management System POC.

This module handles extraction of structured data from ship manuals using LLMs
and Pydantic models for validation and structured output parsing.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
import logging

from pydantic import BaseModel, Field

from .document_loader import load_document, MANUALS_DIR
from .llm_service import LLMService, default_llm_service

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("data_extractor")

# Pydantic models for different categories of extracted data
class GeneralInfo(BaseModel):
    """General information about the ship."""
    nombre: Optional[str] = Field(None, description="Ship name - Usually found in headers or sections about vessel identification")
    puerto: Optional[str] = Field(None, description="Port of registry - Usually preceded by 'PUERTO:' or similar label")
    matricula: Optional[str] = Field(None, description="Registration number - Usually preceded by 'MATRICULA:' or similar label")
    actividad: Optional[str] = Field(None, description="Activity/Purpose - Usually preceded by 'ACTIVIDAD:' or similar label")
    inscripcion: Optional[str] = Field(None, description="Registration details - Usually contains registry information and numbers")
    construido: Optional[str] = Field(None, description="Construction location - Usually preceded by 'CONSTRUIDO:' or similar label")
    año: Optional[str] = Field(None, description="Construction year - Usually preceded by 'AÑO:' and followed by 4-digit year")
    material: Optional[str] = Field(None, description="Hull material - Usually preceded by 'MATERIAL:' or similar label")
    eslora: Optional[str] = Field(None, description="Length overall - Usually preceded by 'ESLORA:' followed by measurement")
    manga: Optional[str] = Field(None, description="Beam - Usually preceded by 'MANGA:' followed by width measurement")
    calado: Optional[str] = Field(None, description="Draft - Usually preceded by 'CALADO:' followed by measurement")
    puntal: Optional[str] = Field(None, description="Depth - Usually preceded by 'PUNTAL:' followed by measurement")
    tonelaje_bruto: Optional[str] = Field(None, description="Gross tonnage - Usually preceded by 'TONELAJE BRUTO:' or similar")

class PropulsionInfo(BaseModel):
    """Propulsion system information."""
    tipo_motor: Optional[str] = Field(None, description="Engine type - Usually found in engine specifications section")
    marca: Optional[str] = Field(None, description="Engine make/brand - Found in engine specifications")
    modelo: Optional[str] = Field(None, description="Engine model - Found in engine specifications")
    numero_serie: Optional[str] = Field(None, description="Engine serial number - Usually preceded by 'Nº SERIE:' or 'S/N:'")
    potencia: Optional[str] = Field(None, description="Engine power - Usually preceded by 'HP:' or power specification")
    rpm: Optional[str] = Field(None, description="Rated RPM - Usually preceded by 'RPM MAX:' or RPM specification")
    combustible: Optional[str] = Field(None, description="Fuel type - Found in fuel system specifications")
    helice: Optional[str] = Field(None, description="Propeller information - Usually preceded by 'HELICE:' or in propulsion section")
    reductora: Optional[str] = Field(None, description="Gearbox information - Usually preceded by 'CAJA REDUCTORA:' or similar")

class CertificateInfo(BaseModel):
    """Certificate information."""
    certificado_navegabilidad: Optional[str] = Field(None, description="Navigability certificate - Found in certificates section, usually with 'NAVEGABILIDAD'")
    certificado_arqueo: Optional[str] = Field(None, description="Tonnage certificate - Found in certificates section, usually with 'ARQUEO:'")
    certificado_seguridad: Optional[str] = Field(None, description="Safety certificate - Found in certificates section, usually with 'SEGURO:' or 'SEGURIDAD'")
    certificado_prevencion_contaminacion: Optional[str] = Field(None, description="Pollution prevention certificate - Found in certificates section")
    certificado_radio: Optional[str] = Field(None, description="Radio certificate - Found in certificates section, might mention 'COMUNICACIONES:'")
    fecha_emision: Optional[str] = Field(None, description="Issue date - Usually preceded by 'EXPEDIDO' or issue date information")
    fecha_expiracion: Optional[str] = Field(None, description="Expiration date - Usually contains expiry date information for certificates")

class ElectricalInfo(BaseModel):
    """Electrical system information."""
    generador: Optional[str] = Field(None, description="Generator information - Found in electrical systems section, may reference 'GENERADORES'")
    voltaje: Optional[str] = Field(None, description="Voltage - Found in electrical specifications, often with voltage units")
    baterias: Optional[str] = Field(None, description="Battery information - Found in electrical systems section under battery descriptions")
    cargador: Optional[str] = Field(None, description="Battery charger - Found in electrical systems regarding charging equipment")
    inversor: Optional[str] = Field(None, description="Inverter information - Found in electrical systems regarding power conversion")
    shore_power: Optional[str] = Field(None, description="Shore power connection - Found in electrical systems regarding external power")

class EngineInfo(BaseModel):
    """Engine specific information from Caterpillar manual."""
    fabricante: Optional[str] = Field(None, description="Engine manufacturer - Usually found in engine specifications section")
    modelo: Optional[str] = Field(None, description="Engine model - Usually found in engine specifications section")
    numero_serie: Optional[str] = Field(None, description="Engine serial number - Usually preceded by 'Nº SERIE:' or similar")
    potencia: Optional[str] = Field(None, description="Engine power - Found in power specifications, often with HP or kW units")
    cilindros: Optional[str] = Field(None, description="Number of cylinders - Found in engine specifications section")
    desplazamiento: Optional[str] = Field(None, description="Engine displacement - Found in engine specifications, usually with volume units")
    sistemas_control: Optional[str] = Field(None, description="Control systems - Found in sections about engine control or management")
    combustible: Optional[str] = Field(None, description="Fuel type - Found in fuel system specifications section")
    refrigeracion: Optional[str] = Field(None, description="Cooling system - Found in cooling system descriptions")
    intervalos_mantenimiento: Optional[str] = Field(None, description="Maintenance intervals - Found in maintenance schedule section")

class BlowerInfo(BaseModel):
    """Blower system information."""
    tipo: Optional[str] = Field(None, description="Blower type - Found in blower system specifications")
    modelo: Optional[str] = Field(None, description="Blower model - Usually found with model number or designation")
    fabricante: Optional[str] = Field(None, description="Manufacturer - Usually found near brand or maker information")
    capacidad: Optional[str] = Field(None, description="Air capacity - Found in performance specifications, often with airflow units")
    presion: Optional[str] = Field(None, description="Pressure - Found in performance specifications with pressure units")
    dimensiones: Optional[str] = Field(None, description="Dimensions - Found in physical specifications section")
    sistema_control: Optional[str] = Field(None, description="Control system - Found in sections about system operation or controls")

# Define extraction schema based on actual manual names
EXTRACTION_SCHEMA = {
    "GFD-Owner Manual-4-1-14.txt": {
        "general_info": GeneralInfo,
        "propulsion": PropulsionInfo,
        "certificates": CertificateInfo,
        "electrical_system": ElectricalInfo
    },
    "MANUAL DE OPERACION Y MANTENIMIENTO CATERPILLAR C4.4.txt": {
        "engine_info": EngineInfo,
        "propulsion": PropulsionInfo
    },
    "DESPIECE_MMAA.txt": {
        "propulsion": PropulsionInfo,
        "electrical_system": ElectricalInfo
    },
    "MN_BLOWERS.txt": {
        "blower_info": BlowerInfo
    },
    "mcw chilles 8.04 L-2164 Manual.txt": {
        "general_info": GeneralInfo,
        "electrical_system": ElectricalInfo
    }
}


class DataExtractor:
    """
    Service for extracting structured data from ship manuals.
    Uses LLMs to extract and validate data according to predefined schemas.
    """
    
    def __init__(self, data_dir: Union[str, Path], llm_provider: str = "gemini"):
        """
        Initialize the data extractor.
        
        Args:
            data_dir (Union[str, Path]): Directory to store extracted data
            llm_provider (str): LLM provider to use ('openai' or 'gemini')
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.llm_provider = llm_provider
        self.llm_service = default_llm_service
        
    def extract_data_from_manual(self, 
                               manual_path: Union[str, Path], 
                               manual_name: str, 
                               categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract structured data from a manual.
        
        Args:
            manual_path (Union[str, Path]): Path to the manual file
            manual_name (str): Name of the manual
            categories (Optional[List[str]]): Specific categories to extract, or None for all
            
        Returns:
            Dict[str, Any]: Extracted structured data
        """
        logger.info(f"Starting extraction for manual: {manual_name}")
        
        # Check if manual has an extraction schema
        if manual_name not in EXTRACTION_SCHEMA:
            logger.error(f"No extraction schema defined for {manual_name}")
            return {"error": f"No extraction schema defined for {manual_name}"}
        
        # Load the manual content
        try:
            logger.info(f"Loading manual content from {manual_path}")
            manual_content = load_document(manual_path)
            logger.info(f"Loaded manual content: {len(manual_content)} characters")
        except Exception as e:
            logger.error(f"Failed to load manual: {str(e)}")
            return {"error": f"Failed to load manual: {str(e)}"}
        
        # Get categories to extract
        schema = EXTRACTION_SCHEMA[manual_name]
        if categories is None:
            categories = list(schema.keys())
            logger.info(f"Using all categories: {categories}")
        else:
            # Filter to only include valid categories
            categories = [cat for cat in categories if cat in schema]
            logger.info(f"Using filtered categories: {categories}")
        
        # Extract data for each category
        extracted_data = {}
        for category in categories:
            logger.info(f"Processing category: {category}")
            model_class = schema[category]
            
            # Extract data for this category
            category_data = self._extract_category_data(
                manual_content=manual_content,
                category=category,
                model_class=model_class
            )
            
            # Store the result
            extracted_data[category] = category_data
            
            # Log the result
            if isinstance(category_data, dict) and "error" in category_data:
                logger.error(f"Error extracting {category}: {category_data['error']}")
            else:
                logger.info(f"Successfully extracted {category}")
        
        return extracted_data
    
    def _extract_category_data(self, 
                              manual_content: str, 
                              category: str, 
                              model_class: Type[BaseModel]) -> Union[Dict[str, Any], BaseModel]:
        """
        Extract data for a specific category from manual content.
        
        Args:
            manual_content (str): The manual content
            category (str): Category name to extract
            model_class (Type[BaseModel]): Pydantic model class for this category
            
        Returns:
            Union[Dict[str, Any], BaseModel]: Extracted data for the category or error dict
        """
        logger.info(f"Beginning extraction for category: {category}")
        
        # Create category-specific extraction instructions
        instructions = f"""
        I need to extract structured information about "{category}" from this technical manual.
        
        Please extract the information following these criteria:
        
        INSTRUCTIONS:
        1. Look for sections or paragraphs specifically dealing with {category}
        2. Identify patterns of labels and values (For example: "LABEL: value")
        3. Pay attention to the document structure and how information is organized
        4. For each requested field, look for related terms and their associated values
        5. Preserve the exact units that appear in the document (meters, kg, etc.)
        6. If you cannot find information for a specific field, indicate null
        
        Focus on the document structure and patterns of data presentation.
        """
        
        try:
            # For extraction tasks, use OpenAI with GPT-4o for best results
            # Temporarily switch provider if needed
            original_provider = self.llm_provider
            
            # Use the LLM service to extract structured data, force OpenAI with GPT-4o
            if self.llm_provider != "openai":
                logger.info(f"Temporarily switching to OpenAI for extraction task")
                self.llm_service = LLMService(provider="openai")
            
            # Use the LLM service to extract structured data
            result = self.llm_service.extract_structured_data(
                content=manual_content,
                output_model=model_class,
                instructions=instructions,
                force_model="gpt-4o"  # Force using GPT-4o for extraction
            )
            
            # Restore original provider if changed
            if self.llm_provider != original_provider:
                logger.info(f"Restoring original provider: {original_provider}")
                self.llm_service = default_llm_service
            
            # Check if an error was returned
            if isinstance(result, dict) and "error" in result:
                return result
            
            # Return the model data as a dict
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Unexpected error extracting {category}: {str(e)}", exc_info=True)
            return {"error": f"Failed to extract data: {str(e)}"}
    
    def save_extracted_data(self, manual_name: str, data: Dict[str, Any]) -> str:
        """
        Save extracted data to a JSON file.
        
        Args:
            manual_name (str): Name of the manual
            data (Dict[str, Any]): Extracted data
            
        Returns:
            str: Path to the saved file
        """
        # Create filename from manual name
        base_name = os.path.splitext(manual_name)[0]
        file_path = self.data_dir / f"{base_name}_extracted.json"
        
        # Save data as JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def load_extracted_data(self, manual_name: str) -> Optional[Dict[str, Any]]:
        """
        Load previously extracted data for a manual.
        
        Args:
            manual_name (str): Name of the manual
            
        Returns:
            Optional[Dict[str, Any]]: Extracted data or None if not found
        """
        base_name = os.path.splitext(manual_name)[0]
        file_path = self.data_dir / f"{base_name}_extracted.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def extract_all_manuals(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract data from all configured manuals.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping manual names to their extracted data
        """
        results = {}
        
        for manual_name in EXTRACTION_SCHEMA.keys():
            manual_path = MANUALS_DIR / manual_name
            if not manual_path.exists():
                results[manual_name] = {"error": f"Manual file not found: {manual_name}"}
                continue
                
            results[manual_name] = self.extract_data_from_manual(
                manual_path=manual_path,
                manual_name=manual_name
            )
            
            # Save the extracted data
            self.save_extracted_data(manual_name, results[manual_name])
            
        return results 