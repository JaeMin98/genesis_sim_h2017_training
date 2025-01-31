import random
from datetime import datetime

random_adjectives = [
    "ancient", "breezy", "crimson", "dazzling", "electric", "epic", "ethereal", "flawless", "frosty", "golden",
    "hidden", "icy", "jagged", "keen", "limitless", "mighty", "mystic", "noble", "ominous", "pristine",
    "quiet", "radiant", "shimmering", "silent", "sparkling", "surreal", "timeless", "tranquil", "twinkling", "unseen",
    "vibrant", "whispering", "wondrous", "zealous", "boundless", "cosmic", "divine", "evergreen", "flowing", "gleaming",
    "harmonic", "illusive", "jubilant", "kaleidoscopic", "luminescent", "majestic", "nebulous", "opulent", "polar", "quivering",
    "roaring", "soaring", "tremendous", "undying", "velvety", "wandering", "xenial", "yearning", "zesty", "amaranthine",
    "bewitching", "cascading", "dreamy", "enigmatic", "foreboding", "glimmering", "haunting", "iridescent", "joyous", "karmic",
    "mesmerizing", "nocturnal", "oceanic", "prismatic", "quintessential", "rustic", "sapphire", "tempestuous", "ultraviolet", "verdant",
    "whimsical", "xanthic", "yonder", "zephyrous", "astral", "blissful", "celestial", "daring", "euphoric", "fiery",
    "galactic", "heavenly", "incandescent", "jovial", "kinetic", "lunar", "magnetic", "nascent", "orbital", "paradoxical",
    "quantum", "resplendent", "stellar", "tidal", "universal", "vortex", "wild", "xeric", "youthful", "glorious"
]

# 2. 명사 100개
random_nouns = [
    "abyss", "arcade", "bard", "blossom", "canyon", "citadel", "cradle", "dawn", "eden", "fabric",
    "fable", "garrison", "gazebo", "glade", "grotto", "haven", "horizon", "jungle", "labyrinth", "lagoon",
    "meadow", "mirage", "nebula", "oasis", "panorama", "paradise", "pinnacle", "quarry", "realm", "reef",
    "riverbank", "sanctum", "silhouette", "skyline", "symphony", "tavern", "temple", "tundra", "utopia", "vacuum",
    "village", "voyage", "waterfall", "wilderness", "xenolith", "yard", "yonderland", "zenith", "azure", "bastion",
    "chamber", "cosmos", "delta", "enclave", "equinox", "festival", "fjord", "forge", "galaxy", "gallery",
    "gateway", "gorge", "grove", "harbor", "hearth", "hive", "inlet", "island", "junction", "kiln",
    "lab", "library", "mansion", "market", "marshland", "mine", "monolith", "monument", "observatory", "opera",
    "orchard", "outpost", "pavilion", "plateau", "portal", "prism", "quagmire", "rainforest", "rapids", "rift",
    "sanctuary", "sculpture", "shipyard", "shrine", "solstice", "spire", "spring", "stronghold", "terrain", "throne",
    "trench", "underworld", "volcano", "waterway", "workshop", "ziggurat", "castle", "glacier", "sands", "shipwreck"
]

def get_random_name():
    """'YYYYMMDD-형용사+명사' 형태의 문자열 생성"""
    date_str = datetime.now().strftime("%Y%m%d%H%M")
    adjective = random.choice(random_adjectives)
    noun = random.choice(random_nouns)
    return f"{date_str}-{adjective}{noun}"