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
    nombre: Optional[str] = Field(None, description="Ship name - Look for 'NOMBRE:', 'Vessel name:', boat name in title sections, or prominently displayed at document start")
    puerto: Optional[str] = Field(None, description="Port of registry - Look for 'PUERTO:', 'Port of Registry:', 'Registrado en:', or in vessel registration/documentation sections")
    matricula: Optional[str] = Field(None, description="Registration number - Look for 'MATRICULA:', 'Registration No.:', 'Matrícula No.:', or numerical identifiers in registration sections")
    actividad: Optional[str] = Field(None, description="Activity/Purpose - Look for 'ACTIVIDAD:', 'Purpose:', 'Vessel use:', keywords like 'DEPORTE', 'comercial', 'recreativo', 'passenger vessel', etc.")
    inscripcion: Optional[str] = Field(None, description="Registration details - Look for inscription references with format like 'INSCRIPTO CON Nº X FOLIO Y LIBRO Z' or 'Registered in the National Registry of Ships'")
    construido: Optional[str] = Field(None, description="Construction location - Look for 'CONSTRUIDO:', 'Built in:', 'Constructed at:', city names, or shipyard locations")
    año: Optional[str] = Field(None, description="Construction year - Look for 'AÑO:', 'Built:', 'Year:', '19XX', '20XX', 4-digit years following construction location")
    material: Optional[str] = Field(None, description="Hull material - Look for 'MATERIAL:', 'Hull material:', 'Construction:', terms like 'fibra de carbono', 'COMPOSITE', 'steel', 'aluminum', 'GRP'")
    eslora: Optional[str] = Field(None, description="Length overall - Look for 'ESLORA:', 'LOA:', 'Length:', 'Length overall:', followed by measurements with units (meters, m, MTS)")
    manga: Optional[str] = Field(None, description="Beam/width - Look for 'MANGA:', 'Beam:', 'Width:', followed by measurements with units (meters, m, MTS)")
    calado: Optional[str] = Field(None, description="Draft - Look for 'CALADO:', 'Draft:', 'Depth:', followed by measurements with units (meters, m, MTS)")
    puntal: Optional[str] = Field(None, description="Depth/height - Look for 'PUNTAL:', 'Depth:', 'Height:', followed by measurements with units (meters, m, MTS)")
    tonelaje_bruto: Optional[str] = Field(None, description="Gross tonnage - Look for 'TONELAJE BRUTO:', 'Gross tonnage:', 'GT:', 'GRT:', followed by numerical values and possibly 'TN', 'tons', 'toneladas'")

class PropulsionInfo(BaseModel):
    """Propulsion system information."""
    tipo_motor: Optional[str] = Field(None, description="Engine type - Look for 'TIPO MOTOR:', 'Engine type:', 'Motor:', terms like 'Diesel', '4-stroke', '4 Tiempos', 'Gasolina', including cylinder counts")
    marca: Optional[str] = Field(None, description="Engine make/brand - Look for 'MARCA:', 'Make:', 'Brand:', manufacturer names like 'Caterpillar', 'Yanmar', 'Volvo', often preceding model numbers")
    modelo: Optional[str] = Field(None, description="Engine model - Look for 'MODELO:', 'Model:', alphanumeric codes like 'C18 Acert', often following manufacturer name")
    numero_serie: Optional[str] = Field(None, description="Engine serial number - Look for 'Nº SERIE:', 'S/N:', 'Serial Number:', 'T2PXXXXX', alphanumeric codes with possible dashes or spaces")
    potencia: Optional[str] = Field(None, description="Engine power - Look for 'POTENCIA:', 'HP:', 'Power:', 'kW:', numerical values followed by units like 'HP', 'kW', 'CV'")
    rpm: Optional[str] = Field(None, description="Rated RPM - Look for 'RPM MAX:', 'RPM:', 'Speed:', numerical values around 1000-3000, sometimes with 'rpm' suffix")
    combustible: Optional[str] = Field(None, description="Fuel type - Look for 'COMBUSTIBLE:', 'Fuel:', 'Fuel type:', terms like 'Diesel', 'Gasoline', 'Gas oil', 'Nafta'")
    helice: Optional[str] = Field(None, description="Propeller information - Look for 'HELICE:', 'Propeller:', 'Prop:', descriptions with diameter, number of blades, pitch, e.g., 'Hélice de paso variable de 1400 mm'")
    reductora: Optional[str] = Field(None, description="Gearbox information - Look for 'CAJA REDUCTORA:', 'Gearbox:', 'Transmission:', brand names like 'Hundested', model numbers, and ratios like '3,96:1'")

class CertificateInfo(BaseModel):
    """Certificate information."""
    certificado_navegabilidad: Optional[str] = Field(None, description="Navigability certificate - Look for 'NAVEGABILIDAD', 'Certificate of Seaworthiness', expiration dates like 'V-MONTH YEAR', may include issuing authority ('URUGUAY')")
    certificado_arqueo: Optional[str] = Field(None, description="Tonnage certificate - Look for 'ARQUEO:', 'Tonnage Certificate:', 'CERTIFICADO DE ARQUEO', issue dates like 'EXPEDIDO DD MMM YYYY'")
    certificado_seguridad: Optional[str] = Field(None, description="Safety certificate - Look for 'SEGURO:', 'SEGURIDAD', 'Safety Certificate:', 'Hull Insurance', expiration dates like 'MONTH YYYY'")
    certificado_prevencion_contaminacion: Optional[str] = Field(None, description="Pollution prevention certificate - Look for 'MARPOL', 'Pollution Prevention', 'Contaminación', 'Environment Protection'")
    certificado_radio: Optional[str] = Field(None, description="Radio certificate - Look for 'COMUNICACIONES:', 'Radio Certificate:', 'URSEC', expiration dates with format 'V-DD MMM YYYY'")
    fecha_emision: Optional[str] = Field(None, description="Issue date - Look for 'EXPEDIDO', 'Issued on:', 'Fecha emisión:', dates following 'EXPEDIDO' or similar terms")
    fecha_expiracion: Optional[str] = Field(None, description="Expiration date - Look for 'Válido hasta:', 'Expires:', 'V-', dates following certificate names, expiration months/years like 'ENERO 2025'")

class ElectricalInfo(BaseModel):
    """Electrical system information."""
    generador: Optional[str] = Field(None, description="Generator information - Look for 'GENERADOR:', 'Generator:', descriptions of number and type like 'Dos generadores de corriente alterna', brand names, models, power ratings 'kW', 'kVA'")
    voltaje: Optional[str] = Field(None, description="Voltage - Look for 'VOLTAJE:', 'Voltage:', 'Sistema eléctrico:', voltage values with units like 'V', 'volts', phrases describing voltage systems '24 volts de corriente continua'")
    baterias: Optional[str] = Field(None, description="Battery information - Look for 'BATERIAS:', 'Batteries:', descriptions of battery types, capacities with 'Ah', banks, quantities, brands like 'Mastervolt'")
    cargador: Optional[str] = Field(None, description="Battery charger - Look for 'CARGADOR:', 'Charger:', 'Battery charger:', brand names, model numbers, capacities with 'A', voltage specs")
    inversor: Optional[str] = Field(None, description="Inverter information - Look for 'INVERSOR:', 'Inverter:', brand names like 'VICTRON', models, power ratings with 'W' or 'kW', voltage input/output specs")
    shore_power: Optional[str] = Field(None, description="Shore power connection - Look for 'TOMA DE TIERRA:', 'Shore power:', 'Shore connection:', descriptions of shore power systems, amperage ratings, voltage specifications")

class EngineInfo(BaseModel):
    """Engine specific information from Caterpillar manual."""
    fabricante: Optional[str] = Field(None, description="Engine manufacturer - Look for 'FABRICANTE:', 'Manufacturer:', 'Make:', brand name usually at the beginning of specifications, e.g., 'Caterpillar'")
    modelo: Optional[str] = Field(None, description="Engine model - Look for 'MODELO:', 'Model:', alphanumeric codes like 'C4.4', usually prominently displayed in title or specifications")
    numero_serie: Optional[str] = Field(None, description="Engine serial number - Look for 'Nº SERIE:', 'S/N:', 'Serial Number:', 'Engine Number:', followed by alphanumeric codes, often with prefixes")
    potencia: Optional[str] = Field(None, description="Engine power - Look for 'POTENCIA:', 'Power:', 'Rating:', numerical values with units like 'kW', 'HP', 'BHP', in performance or specifications tables")
    cilindros: Optional[str] = Field(None, description="Number of cylinders - Look for 'CILINDROS:', 'Cylinders:', 'No. of cylinders:', numerical values often with configuration like 'in-line', 'V arrangement'")
    desplazamiento: Optional[str] = Field(None, description="Engine displacement - Look for 'DESPLAZAMIENTO:', 'Displacement:', 'Cubic capacity:', numerical values with units like 'L', 'cc', 'cubic inches'")
    sistemas_control: Optional[str] = Field(None, description="Control systems - Look for 'SISTEMAS DE CONTROL:', 'Control system:', descriptions of engine management systems, electronic controls, ECU references")
    combustible: Optional[str] = Field(None, description="Fuel type - Look for 'COMBUSTIBLE:', 'Fuel type:', 'Fuel system:', descriptions like 'Diesel', 'Injection system', fuel delivery methods")
    refrigeracion: Optional[str] = Field(None, description="Cooling system - Look for 'REFRIGERACION:', 'Cooling system:', 'Cooling:', descriptions of water-cooling, air-cooling, heat exchangers")
    intervalos_mantenimiento: Optional[str] = Field(None, description="Maintenance intervals - Look for 'INTERVALOS DE MANTENIMIENTO:', 'Maintenance schedule:', 'Service intervals:', hour-based intervals, maintenance tables")

class BlowerInfo(BaseModel):
    """Blower system information."""
    tipo: Optional[str] = Field(None, description="Blower type - Look for 'TIPO:', 'Type:', classification terms like 'Centrifugal', 'Axial', 'Reversible', 'Extractor', 'Ventilador'")
    modelo: Optional[str] = Field(None, description="Blower model - Look for 'MODELO:', 'Model:', alphanumeric designations, series numbers, product codes like 'ELL/AP 565'")
    fabricante: Optional[str] = Field(None, description="Manufacturer - Look for 'FABRICANTE:', 'Manufacturer:', 'Make:', 'Brand:', company names like 'GIANNESCHI', often with logos or headers")
    capacidad: Optional[str] = Field(None, description="Air capacity - Look for 'CAPACIDAD:', 'Capacity:', 'Airflow:', numerical values with units like 'm³/h', 'CFM', 'L/min' in specifications or performance tables")
    presion: Optional[str] = Field(None, description="Pressure - Look for 'PRESION:', 'Pressure:', 'Static pressure:', numerical values with units like 'Pa', 'mmH2O', 'bar' in performance specifications")
    dimensiones: Optional[str] = Field(None, description="Dimensions - Look for 'DIMENSIONES:', 'Dimensions:', 'Size:', measurements with units like 'mm', physical size specifications, often in format 'LxWxH'")
    sistema_control: Optional[str] = Field(None, description="Control system - Look for 'SISTEMA DE CONTROL:', 'Control system:', descriptions of switches, sensors, speed controllers, automation systems related to blower operation")

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
        
        # Always use OpenAI for extraction tasks, regardless of default provider
        self.extraction_provider = "openai"
        
        # Define model tiers for different extraction tasks
        self.model_tiers = {
            # Tier 1: Fast, lightweight scanning (lowest token cost)
            "scan": {
                "openai": "gpt-3.5-turbo", 
                "gemini": "gemini-2.0-flash-lite"
            },
            # Tier 2: General extraction (medium token cost)
            "extract": {
                "openai": "gpt-4o-mini",
                "gemini": "gemini-2.0-flash"
            },
            # Tier 3: Complex extraction (highest token cost)
            "complex": {
                "openai": "gpt-4o",
                "gemini": "gemini-2.0-pro" # Updated to use the most powerful Gemini model
            }
        }
        
        # Extraction cache to avoid redundant work
        self._cache = {}

    def _get_llm_for_tier(self, tier: str) -> LLMService:
        """
        Get an LLM service instance for the specified tier.
        
        Args:
            tier (str): The tier level ('scan', 'extract', or 'complex')
            
        Returns:
            LLMService: LLM service configured for the specified tier
        """
        # Default to complex tier if invalid tier specified
        if tier not in self.model_tiers:
            tier = "complex"
            
        # Always use OpenAI for extraction tasks
        provider = self.extraction_provider
        model = self.model_tiers[tier].get(provider)
        
        # Return a new LLM service instance configured for this tier
        return LLMService(provider=provider, model_name=model)

    def extract_data_from_manual(self, 
                               manual_path: Union[str, Path], 
                               manual_name: str, 
                               categories: Optional[List[str]] = None,
                               force_refresh: bool = False) -> Dict[str, Any]:
        """
        Extract structured data from a manual.
        
        Args:
            manual_path (Union[str, Path]): Path to the manual file
            manual_name (str): Name of the manual
            categories (Optional[List[str]]): Specific categories to extract, or None for all
            force_refresh (bool): Whether to force extraction even if data is cached
            
        Returns:
            Dict[str, Any]: Extracted structured data
        """
        # Check if we already have cached results and aren't forcing a refresh
        cache_key = f"{manual_name}:{','.join(categories or [])}"
        if cache_key in self._cache and not force_refresh:
            logger.info(f"Using cached extraction results for {manual_name}")
            return self._cache[cache_key]
        
        logger.info(f"Starting extraction for manual: {manual_name} using OpenAI models")
        
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
            
            # Extract data for this category using staged approach
            category_data = self._extract_category_data_staged(
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
        
        # Cache the results
        self._cache[cache_key] = extracted_data
        
        return extracted_data
    
    def _find_relevant_sections(self, manual_content: str, category: str) -> str:
        """
        Find sections of the manual relevant to a specific category to reduce token usage.
        
        Args:
            manual_content (str): The complete manual content
            category (str): The category to find relevant sections for
            
        Returns:
            str: Concatenated relevant sections from the manual
        """
        # Create a dictionary of category to related keywords
        category_keywords = {
            "general_info": [
                "general", "information", "vessel", "ship", "datos", "generales", 
                "información", "buque", "barco", "nombre", "puerto", "matricula", 
                "eslora", "manga", "calado", "tonelaje", "introduction", "introducción",
                "características", "características generales", "dimensions", "dimensiones",
                "specifications", "especificaciones", "registration", "registro", "construction",
                "hull", "material", "casco", "fibra", "composite", "informacion general",
                "vessel particulars", "particulars", "inscripción", "desplazamiento", "año"
            ],
            "propulsion": [
                "propulsion", "engine", "motor", "propulsión", "caterpillar", 
                "rpm", "power", "potencia", "fuel", "combustible", "helice", 
                "propeller", "gearbox", "reductora", "main engine", "motor principal",
                "línea de eje", "shaft", "transmission", "transmisión", "diesel",
                "cilindros", "cylinders", "horsepower", "hp", "kw", "serie", "serial number",
                "número de serie", "velocidad", "speed", "paso variable", "variable pitch"
            ],
            "certificates": [
                "certificate", "certification", "certificado", "navegabilidad", 
                "arqueo", "seguridad", "contaminacion", "radio", "expedido", 
                "expiracion", "caducidad", "validez", "URSEC", "RINA", "class certificate",
                "tonnage", "safety", "pollution", "environment", "registration certificate",
                "certificate of registry", "expiry date", "fecha de caducidad", "renewal",
                "renovación", "authority", "autoridad", "documento", "document", "issued",
                "emitido", "regulatory", "regulatorio", "compliance", "cumplimiento"
            ],
            "electrical_system": [
                "electrical", "electric", "eléctrico", "eléctrica", 
                "generator", "generador", "voltage", "voltaje", "batería", 
                "battery", "inverter", "inversor", "shore power", "energía",
                "power system", "sistema eléctrico", "ampere", "amp", "corriente",
                "current", "breaker", "interruptor", "panel", "transformador", "transformer",
                "charger", "cargador", "alternator", "alternador", "power supply",
                "distribution", "distribución", "connection", "conexión", "AC", "DC",
                "alterna", "continua", "ASEA", "mastervolt", "victron"
            ],
            "engine_info": [
                "engine", "motor", "caterpillar", "manufacturer", "fabricante", 
                "model", "modelo", "serie", "serial", "potencia", "power", 
                "cylinder", "cilindro", "cooling", "refrigeración", "displacement",
                "desplazamiento", "fuel system", "sistema de combustible", "injection",
                "inyección", "maintenance", "mantenimiento", "service", "servicio",
                "specifications", "especificaciones", "operation", "operación",
                "engine control", "control del motor", "torque", "par", "compression",
                "compresión", "temperature", "temperatura", "oil", "aceite"
            ],
            "blower_info": [
                "blower", "ventilador", "soplador", "air", "aire", "fan", 
                "pressure", "presión", "capacity", "capacidad", "extractor",
                "intake", "admisión", "exhaust", "escape", "flow", "flujo", 
                "ventilation", "ventilación", "CFM", "m3/h", "static pressure",
                "presión estática", "centrifugal", "centrífugo", "axial", "reversible",
                "gianneschi", "dimension", "dimensión", "install", "instalación",
                "mounting", "montaje", "duct", "ducto", "control", "controller"
            ]
        }
        
        # Get keywords for the requested category
        keywords = category_keywords.get(category, [])
        if not keywords:
            # If no specific keywords defined, return the full content
            logger.warning(f"No keywords defined for category {category}")
            return manual_content
        
        # Split the content into lines
        lines = manual_content.split('\n')
        
        # Mark lines containing keywords
        relevant_line_indices = []
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for keyword in keywords:
                if keyword.lower() in line_lower:
                    # Add this line index and some context around it
                    context_range = 25  # Lines of context before and after (increased from 20)
                    start_idx = max(0, i - context_range)
                    end_idx = min(len(lines), i + context_range)
                    relevant_line_indices.extend(range(start_idx, end_idx))
                    break
        
        # Remove duplicates and sort
        relevant_line_indices = sorted(set(relevant_line_indices))
        
        # Group consecutive indices to identify sections
        sections = []
        current_section = []
        for idx in relevant_line_indices:
            if not current_section or idx == current_section[-1] + 1:
                current_section.append(idx)
            else:
                sections.append(current_section)
                current_section = [idx]
        if current_section:
            sections.append(current_section)
        
        # Extract sections with a bit more context
        relevant_content = []
        for section in sections:
            start = max(0, section[0] - 10)  # Increased context from 5 to 10
            end = min(len(lines), section[-1] + 10)  # Increased context from 5 to 10
            section_text = '\n'.join(lines[start:end])
            relevant_content.append(section_text)
        
        # Combine all relevant sections
        combined_content = '\n\n[...]\n\n'.join(relevant_content)
        
        # If we didn't find anything, return a portion of the document
        if not combined_content:
            logger.warning(f"No relevant sections found for {category}")
            # Return more of the document - beginning, middle and end
            head_lines = 200  # Increased from 100
            middle_start = max(0, len(lines) // 2 - 100)
            middle_end = min(len(lines), len(lines) // 2 + 100)
            tail_lines = 200  # Increased from 100
            combined_content = '\n'.join(lines[:head_lines]) + '\n\n[...]\n\n' + \
                              '\n'.join(lines[middle_start:middle_end]) + '\n\n[...]\n\n' + \
                              '\n'.join(lines[-tail_lines:])
        
        logger.info(f"Extracted {len(combined_content)} characters of relevant content for {category} from {len(manual_content)} total")
        return combined_content

    def _extract_category_data_staged(self, 
                                    manual_content: str, 
                                    category: str, 
                                    model_class: Type[BaseModel]) -> Union[Dict[str, Any], BaseModel]:
        """
        Extract data for a category using a staged approach with different model tiers.
        Starts with lightweight models and escalates to more powerful ones when needed.
        
        Args:
            manual_content (str): The manual content
            category (str): Category name to extract
            model_class (Type[BaseModel]): Pydantic model class for this category
            
        Returns:
            Union[Dict[str, Any], BaseModel]: Extracted data for the category or error dict
        """
        logger.info(f"Beginning staged extraction for category: {category}")
        
        # Stage 1: Use scan tier to find relevant sections
        scan_llm = self._get_llm_for_tier("scan")
        
        # Use the _find_relevant_sections method to get relevant content
        relevant_content = self._find_relevant_sections(manual_content, category)
        
        # Create standard instructions
        instructions = self._create_extraction_instructions(category)
        
        # Track errors for potential retry with more powerful model
        extraction_error = None
        empty_fields_count = 0
        
        # Attempt extraction with progressively more powerful models
        
        # Stage 2: Try with mid-tier model first (good balance of performance vs. cost)
        try:
            logger.info(f"Attempting extraction with 'extract' tier model for {category}")
            extract_llm = self._get_llm_for_tier("extract")
            
            # Use the LLM service to extract structured data
            result = extract_llm.extract_structured_data(
                content=relevant_content,
                output_model=model_class,
                instructions=instructions,
                simplified=True  # Use simplified prompts for mid-tier models
            )
            
            # Check if an error was returned
            if isinstance(result, dict) and "error" in result:
                extraction_error = result["error"]
                logger.warning(f"Extract tier failed with: {extraction_error}. Will escalate to complex tier.")
            else:
                # Count empty fields to check extraction quality
                empty_fields = sum(1 for v in result.model_dump().values() if v is None)
                empty_fields_count = empty_fields
                field_count = len(result.model_dump())
                
                # If more than 70% of fields are empty, might need a more powerful model
                if empty_fields > field_count * 0.7:
                    logger.warning(f"Extraction quality low: {empty_fields}/{field_count} empty fields. Will try complex model.")
                else:
                    # Good enough extraction quality
                    return result.model_dump()
                
        except Exception as e:
            extraction_error = str(e)
            logger.warning(f"Extract tier extraction error: {extraction_error}. Will escalate to complex tier.")
        
        # Stage 3: Fall back to the most powerful model for challenging extractions
        try:
            logger.info(f"Attempting extraction with 'complex' tier model for {category}")
            complex_llm = self._get_llm_for_tier("complex")
            
            # Enhanced instructions for difficult cases
            enhanced_instructions = instructions + """
            
            ADDITIONAL GUIDANCE FOR CHALLENGING EXTRACTION:
            - This content has been identified as especially difficult to extract accurately
            - Pay special attention to contextual clues and implied information
            - Look for information across different sections that may relate to the same fields
            - Use logical inference based on technical domain knowledge when direct matches aren't found
            """
            
            # Use the most powerful LLM for complex extraction
            result = complex_llm.extract_structured_data(
                content=relevant_content,
                output_model=model_class,
                instructions=enhanced_instructions,
                simplified=False  # Use full prompting for complex models
            )
            
            # Check if an error was returned
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Complex tier extraction also failed: {result['error']}")
                return result
                
            # Return the model data as a dict
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Unexpected error in complex tier extraction: {str(e)}", exc_info=True)
            return {"error": f"Failed to extract data after multiple attempts: {str(e)}"}
    
    def _create_extraction_instructions(self, category: str) -> str:
        """
        Create detailed extraction instructions for a specific category.
        
        Args:
            category (str): The category being extracted
            
        Returns:
            str: Formatted instructions for the LLM
        """
        return f"""
        I need to extract structured information about "{category}" from this technical manual.
        
        Please extract the information following these criteria:
        
        INSTRUCTIONS:
        1. Look for sections or paragraphs specifically dealing with {category}
        2. Identify patterns of labels and values in various formats:
           - "LABEL: value" (Spanish or English labels)
           - "value (LABEL)" format
           - Tables with headers and values
           - Lists with specification details
        3. Pay close attention to document structure - information may be in:
           - Specification tables
           - Technical data sections
           - Titled paragraphs
           - Bulleted lists
           - Headers and footers
        4. Look for numbers with their units (meters, kg, HP, kW, etc.)
        5. For serial numbers and identifiers, look for patterns like:
           - "S/N: XXXXX" 
           - "Nº SERIE: XXXXX"
           - Alphanumeric codes often near equipment descriptions
        6. Extract quantities when available (e.g., "2x generators" should include the quantity)
        7. If a field has multiple related values, combine them in a meaningful way
        8. If information is not found, leave the field as null
        
        Focus on accurate extraction of values exactly as they appear in the document.
        
        Note: This content contains only the parts of the manual most likely to be relevant 
        to {category}. Some sections may be omitted (marked with [...]).
        """
    
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