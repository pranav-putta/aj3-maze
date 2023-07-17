# evaluate an agent
import imageio
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFont
import cv2


def frames_to_mp4(frames, filename):
    """Save a list of frames as an mp4 file."""
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 3, videodims)
    # draw stuff that goes on every frame here
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()


def evaluate_and_store_mp4(env, agent, filename):
    """Evaluate an agent and store the result as an mp4 file."""
    env.reset()
    frames = []
    done = False
    i = 0

    state = env.reset()
    hx = None
    while not done:
        state = torch.tensor(np.array(state))[None,]
        policy_output = agent.act(state, hx=hx)
        state, _, done, infos = env.step(policy_output.action)
        hx = policy_output.hidden_state

        img = env.render(mode='rgb_array')
        img = Image.fromarray(img)

        # Add padding to the image
        font_size = 32
        font = ImageFont.truetype("arial.ttf", size=font_size)  # Replace "arial.ttf" with your desired font file
        padding = 50
        if infos['valid'] and not infos['success']:
            padding_color = (0, 0, 0)  # White color, you can change it to your desired color
        elif infos['valid'] and infos['success']:
            padding_color = (0, 255, 0)
        else:
            padding_color = (255, 0, 0)
        padded_image = ImageOps.expand(img, border=padding, fill=padding_color)
        draw = ImageDraw.Draw(padded_image)
        text = "Action: {}".format(policy_output.action.item())  # Replace "action" with your desired action text
        text_width, text_height = draw.textsize(text, font=font)
        x = (padded_image.width - text_width) // 2
        y = padded_image.height - padding - text_height * 2

        # Overlay the text on the image
        draw.text((x, y), text, fill=(255, 255, 255), font=font)
        frames.append(padded_image)

    env.close()
    frames_to_mp4(frames, filename)
