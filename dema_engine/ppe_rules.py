"""
PPE Safety Rules and Standard Definitions for Substation Hazard Compliance.
Supports both YOLO trained models and SSD-VGG16 models.
"""

# Target PPE Classes
TARGET_CLASSES = ["Helmet", "Gloves", "Footwear", "Safety Vest", "Mask"]

# Mapping of class aliases and non-compliance aliases to standard names
CLASS_ALIASES = {
    "hardhat": "Helmet",
    "helmet": "Helmet",
    "hat": "Helmet",
    "no-hardhat": "NO-Helmet",
    "no-helmet": "NO-Helmet",
    "gloves": "Gloves",
    "glove": "Gloves",
    "no-gloves": "NO-Gloves",
    "footwear": "Footwear",
    "boots": "Footwear",
    "shoes": "Footwear",
    "safety shoes": "Footwear",
    "leather boots": "Footwear",
    "no-footwear": "NO-Footwear",
    "no-boots": "NO-Footwear",
    "safety vest": "Safety Vest",
    "vest": "Safety Vest",
    "no-safety vest": "NO-Safety Vest",
    "no-vest": "NO-Safety Vest",
    "mask": "Mask",
    "no-mask": "NO-Mask",
    "person": "Person",
    "human": "Person",
    "worker": "Person",
    "safety cone": "Safety Cone",
    "machinery": "Machinery",
    "vehicle": "Vehicle",
}

# Hazard Levels & Mandatory PPE Requirements (NFPA 70E / Substation Standard)
HAZARD_LEVEL_REQUIREMENTS = {
    "Level_1_Standard": {
        "description": "Standard Substation Inspection",
        "required": {"Helmet", "Footwear"},
        "optional": {"Gloves", "Safety Vest", "Mask"}
    },
    "Level_2_High_Risk": {
        "description": "High Voltage Work / Electrical Maintenance",
        "required": {"Helmet", "Gloves", "Footwear"},
        "optional": {"Safety Vest", "Mask"}
    }
}

def normalize_class_name(class_name: str) -> str:
    """Normalizes class names into standard names."""
    key = class_name.strip().lower()
    return CLASS_ALIASES.get(key, class_name)
