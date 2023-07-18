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
    0: (160, 160, 160),  # obstacle
    1: (224, 224, 224),  # free
    2: (51, 153, 255),  # agent
    3: (51, 255, 51),  # goal
    4: (102, 0, 204),  # button
    5: (255, 0, 255),  # interruption
    6: (0, 102, 102),  # box
    7: (255, 0, 0),  # lava
    8: (0, 0, 255),  # water
}
