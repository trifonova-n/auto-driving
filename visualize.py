import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_video(video_iter):
    fig = plt.figure(figsize=(10 ,10))
    frames = []
    for sample in video_iter:
        im = plt.imshow( sample[0], animated=True)
        frames.append([im])

    plt.close(fig)
    return animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
