"""
PPE Safety Rules and Standard Definitions for Substation Hazard Compliance.
Focuses specifically on Helmet, Gloves, and Footwear as requested.
"""

# Mandatory Target PPE Classes
TARGET_CLASSES = ["Helmet", "Gloves", "Footwear"]

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
        "required": {"Helmet", "Gloves", "Footwear"},
        "optional": set()
    }
}

def normalize_class_name(class_name: str) -> str:
    """Normalizes class names into standard names."""
    key = class_name.strip().lower()
    return CLASS_ALIASES.get(key, class_name)
