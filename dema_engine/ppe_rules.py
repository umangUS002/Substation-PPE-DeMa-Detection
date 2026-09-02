"""
PPE Safety Rules and Standard Definitions for Substation Hazard Compliance.
Supports both YOLO trained models and SSD-VGG16 models.
"""

# Target PPE Classes
TARGET_CLASSES = ["Helmet", "Footwear", "Gloves"]

# Mapping of class aliases and non-compliance aliases to standard names
CLASS_ALIASES = {
    "hardhat": "Helmet",
    "helmet": "Helmet",
    "hat": "Helmet",
    "no-hardhat": "NO-Helmet",
    "no-helmet": "NO-Helmet",
    "footwear": "Footwear",
    "boots": "Footwear",
    "shoes": "Footwear",
    "safety shoes": "Footwear",
    "leather boots": "Footwear",
    "no-footwear": "NO-Footwear",
    "no-boots": "NO-Footwear",
    "gloves": "Gloves",
    "glove": "Gloves",
    "no-gloves": "NO-Gloves",
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
        "optional": {"Gloves"}
    },
    "Level_2_High_Risk": {
        "description": "High Voltage Work / Electrical Maintenance",
        "required": {"Helmet", "Footwear", "Gloves"},
        "optional": set()
    }
}

def normalize_class_name(class_name: str) -> str:
    """Normalizes class names into standard names."""
    key = class_name.strip().lower()
    return CLASS_ALIASES.get(key, class_name)
