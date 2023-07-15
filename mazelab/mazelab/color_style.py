from dataclasses import dataclass


@dataclass
class DeepMindColor:
    obstacle = (160, 160, 160)
    free = (224, 224, 224)
    agent = (51, 153, 255)
    goal = (51, 255, 51)
    button = (102, 0, 204)
    interruption = (255, 0, 255)
    box = (0, 102, 102)
    lava = (255, 0, 0)
    water = (0, 0, 255)


object_colors = {
    "Apricot": "#FBB982",
    "Aqua": "#00FFFF",
    "Azure": "#007FFF",
    "Blush": "#DE5D83",
    "Coral": "#FF7F50",
    "Crimson": "#DC143C",
    "Emerald": "#50C878",
    "Goldenrod": "#DAA520",
    "Lavender": "#E6E6FA",
    "Magenta": "#FF00FF",
    "Mint": "#98FF98",
    "Peach": "#FFDAB9",
    "Periwinkle": "#CCCCFF",
    "Rose": "#FF007F",
    "Saffron": "#F4C430",
    "Sapphire": "#0F52BA",
    "Scarlet": "#FF2400",
    "Silver": "#C0C0C0",
    "Teal": "#008080",
    "Turquoise": "#40E0D0"
}
