import os
import imageio


def count_frames(movie_path):
    n_frames = None
    if os.path.exists(movie_path):
        with imageio.get_reader(movie_path) as mov_reader:
            n_frames = mov_reader.count_frames()

    return n_frames


def get_mov_res(movie_path):
    x = None
    y = None
    if os.path.exists(movie_path):
        with imageio.get_reader(movie_path) as mov_reader:
            (x, y) = mov_reader.get_meta_data()['size']

    return (x, y)


def get_mov_fps(movie_path):
    fps = None
    if os.path.exists(movie_path):
        with imageio.get_reader(movie_path) as mov_reader:
            fps = mov_reader.get_meta_data()['fps']

    return fps