# evaluate an agent
import imageio
import numpy as np
from PIL import Image
import cv2


def frames_to_mp4(frames, filename):
    """Save a list of frames as an mp4 file."""
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video = cv2.VideoWriter(filename, fourcc, 60, videodims)
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
    while not done:
        frames.append(Image.fromarray(env.render(mode='rgb_array')))
        action, dist = agent.act(state[None,])
        state, _, done, _ = env.step(action)

        img = env.render(mode='rgb_array')
        frames.append(img)

    env.close()
    frames_to_mp4(frames, filename)
