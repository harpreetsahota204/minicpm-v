
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional 

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

- Include all relevant elements that match the user's request
- For UI elements, include their function when possible (e.g., "Login Button" rather than just "Button")
- If many similar elements exist, prioritize the most prominent or relevant ones

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and detect.
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label",
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image, but limited to one or two word responses
- The response should be a list of classifications
"""

DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point identify the key point and provide a contextually appropriate label and always return your response wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "label": "descriptive label for the point"
        }
    ]
}
```

The JSON should contain points in pixel coordinates [x,y] format, where:
- x is the horizontal center coordinate of the visual element
- y is the vertical center coordinate of the visual element
- Include all relevant elements that match the user's request
- You can point to multiple visual elements
"""

DEFAULT_OCR_SYSTEM_PROMPT = """You are an OCR assistant. Your task is to identify and extract all visible text from the image provided. Preserve the original formatting as closely as possible, including:

- Line breaks and paragraphs  
- Headings and subheadings  
- Any tables, lists, bullet points, or numbered items  
- Special characters, spacing, and alignment  

Output strictly the extracted text in Markdown format, reflecting the layout and structure of the original image. Do not add commentary, interpretation, or summarization—only return the raw text content with its formatting.

Respond with 'No Text' if there is no text in the provided image.
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

OPERATIONS = {
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class MiniCPM_V(SamplesMixin, Model):
    """A FiftyOne model for running MiniCPM-V 4.5 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }
        
        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model.eval()

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        This method handles JSON within markdown code blocks (```json) or raw JSON strings.
        
        Args:
            s: Raw string output from the model containing JSON
                
        Returns:
            Dict: The parsed JSON content 
            None: If parsing fails
            Original input: If input is not a string
        """
        # Return non-string inputs as-is
        if not isinstance(s, str):
            return s
        
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            try:
                # Split on markdown markers and take JSON content
                json_str = s.split("```json")[1].split("```")[0].strip()
            except:
                json_str = s
        else:
            json_str = s
            
        # Attempt to parse the JSON string
        try:
            parsed_json = json.loads(json_str)
            return parsed_json
        except:
            # Log parsing failures for debugging
            logger.debug(f"Failed to parse JSON: {json_str[:200]}")
            return None
    
    def _parse_box_tags(self, text: str) -> List[Dict]:
        """Parse bounding boxes from <ref>object</ref><box>x1 y1 x2 y2</box> format.
        
        Args:
            text: Model output containing ref and box tags
            
        Returns:
            List of dictionaries with bbox coordinates and labels
        """
        import re
        
        detections = []
        
        # Pattern to match <ref>label</ref><box>x1 y1 x2 y2</box>
        pattern = r'<ref>([^<]+)</ref><box>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)</box>'
        
        matches = re.findall(pattern, text)
        
        for match in matches:
            label = match[0].strip()
            x1, y1, x2, y2 = match[1:5]
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': label if label else "object"
            })
        
        return detections

    def _to_detections(self, text: str) -> fo.Detections:
        """Convert detection text output to FiftyOne Detections.
        
        Parses the <ref>object</ref><box>x1 y1 x2 y2</box> format and converts
        to FiftyOne's format with coordinate normalization from 0-1000 to 0-1 range.
        
        Args:
            text: Model output string containing ref and box tags
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
        """
        detections = []
        
        # Parse the box tags from the text
        parsed_boxes = self._parse_box_tags(text)
        
        # Process each bounding box
        for box in parsed_boxes:
            try:
                bbox = box['bbox']
                label = box['label']
                
                # Convert coordinates to float and normalize from 0-1000 to 0-1 range
                x1_norm, y1_norm, x2_norm, y2_norm = map(float, bbox)
                
                x = x1_norm / 1000.0  # Left coordinate (0-1)
                y = y1_norm / 1000.0  # Top coordinate (0-1)
                w = (x2_norm - x1_norm) / 1000.0  # Width (0-1)
                h = (y2_norm - y1_norm) / 1000.0  # Height (0-1)
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h]
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications.
        
        Processes classification labels into FiftyOne's format.
        
        Args:
            classes: Classification results, either:
                - List of classification dictionaries
                - Dictionary containing classifications
                
        Returns:
            fo.Classifications: FiftyOne Classifications object containing all results
            
        Example input:
            {
                "classifications": [{"label": "cat"}, {"label": "animal"}]
            }
        """
        classifications = []
        
        # Handle nested dictionary structures
        if isinstance(classes, dict):
            classes = classes.get("classifications", classes)
            if isinstance(classes, dict):
                classes = next((v for v in classes.values() if isinstance(v, list)), classes)
        
        # Process each classification
        for cls in classes:
            try:
                # Create FiftyOne Classification object
                classification = fo.Classification(
                    label=str(cls["label"])
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _to_keypoints(self, points: Union[List[Dict], Dict]) -> fo.Keypoints:
        """Convert keypoint results to FiftyOne Keypoints.
        
        Converts keypoints from model's 0-1000 normalized coordinates to FiftyOne's 0-1 range.
        
        Args:
            points: Keypoint results, either:
                - List of point dictionaries
                - Dictionary containing 'keypoints' list
                
        Returns:
            fo.Keypoints object containing the converted keypoint annotations
            with coordinates normalized to [0,1] x [0,1] range
        
        Expected input format:
        {
            "keypoints": [
                {"point_2d": [100, 200], "label": "person's head"},
                {"point_2d": [300, 400], "label": "dog's nose"}
            ]
        }
        where coordinates are in 0-1000 range.
        """
        keypoints = []
        
        # Handle nested dictionary structures
        if isinstance(points, dict):
            points = points.get("keypoints", points)
            if isinstance(points, dict):
                points = next((v for v in points.values() if isinstance(v, list)), points)
        
        # Ensure we're working with a list
        points = points if isinstance(points, list) else [points]
        
        for point in points:
            try:
                # Get coordinates from point_2d field (normalized to 1000)
                x, y = point["point_2d"]
                x = float(x)
                y = float(y)
                
                # Convert from 0-1000 range to 0-1 range for FiftyOne
                normalized_point = [
                    x / 1000.0,
                    y / 1000.0
                ]
                
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Classifications, fo.Keypoints, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/keypoint/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")

        msgs = [{'role': 'user', 'content': [image,  prompt]}]

        generation_config = dict(
            max_new_tokens=4096, 
            do_sample=False, 
            pad_token_id=self.tokenizer.eos_token_id,
            )

        output_text = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer, 
            system_prompt = self.system_prompt,
            generation_config=generation_config)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "ocr":
            return output_text.strip()
        elif self.operation == "detect":
            return self._to_detections(output_text)
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)