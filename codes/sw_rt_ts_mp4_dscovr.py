# -*- coding: utf-8 -*-

import datetime
import glob as glob
import os
import time

import imageio as iio
import numpy as np

from pathlib import Path


def gif_maker(
    file_list, vid_name, mode="I", skip_rate=10, vid_type="mp4", duration=0.05, fps=25
):
    """
    Make a gif from a list of images.

    Parameters
    ----------
    file_list : list
        List of image files.
    vid_name : str
        Name of the gif file.
    mode : str, optional
        Mode of the gif. The default is "I".
    skip_rate : int, optional
        Skip rate of the gif. The default is 10.
    vid_type : str, optional
        Type of the video. The default is "mp4".
    duration : float, optional
        Duration for which each image is displayed in gif. The default is 0.05.
    fps : int, optional
        Frames per second for mp4 video. The default is 25.

    Raises
    ------
    ValueError
        If the skip_rate is not an integer.
    ValueError
        If the duration is not a float.
    ValueError
        If the file_list is empty.
    ValueError
        If vid_name is empty.

    Returns
    -------
    None.
    """
    if file_list is None:
        raise ValueError("file_list is None")
    if vid_name is None:
        raise ValueError("vid_name is None. Please provide the name of the gif/video")
    if len(file_list) == 0:
        raise ValueError("file_list is empty")
    # if len(file_list) >= 1501:
    #     # Check if the skip_rate is an integer
    #     if skip_rate != int(skip_rate):
    #         raise ValueError("skip_rate must be an integer")
    #     file_list = file_list[-1500::skip_rate]
    if vid_type == "gif":
        if duration != float(duration):
            raise ValueError("duration must be a float")
    if vid_type == "mp4":
        if fps != int(fps):
            raise ValueError("Frame rate (fps) must be an integer")

    count = 0
    if vid_type == "gif":
        with iio.get_writer(vid_name, mode=mode, duration=duration) as writer:
            for filename in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                try:
                    img = iio.imread(filename)
                    writer.append_data(img)
                except Exception as e:
                    print(e)
                    pass
    elif vid_type == "mp4":
        with iio.get_writer(vid_name, mode=mode, fps=fps) as writer:
            for filename in file_list:
                count += 1
                print(f"Processing image {count} of {len(file_list)}")
                try:
                    img = iio.imread(filename)
                    writer.append_data(img)
                except Exception as e:
                    print(e)
                    pass
    writer.close()

    # Print that the video is created along with the time of creation in UTC
    print(
        f"Video created at (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
    )

    # Copy the file to a specific location
    os.system(f"cp {vid_name} ~/Dropbox/rt_sw/")


def make_gifs(
    number_of_files=120, vid_type="mp4", image_path="images/", gif_path="videos/"
):
    """
    Make gifs for the last n days. Default is 120 days, averaged over the last 07 days.

    Parameters
    ----------
    number_of_files : int, optional
        Number of files to be considered for plotting the gif. The default is 120.
    vid_type : str, optional
        Type of the video. The default is "mp4". Other option is "gif".
    image_path : str, optional
        Path of the location of images. The default is "../images/".
    gif_path : str, optional
        Path to save the gif. The default is "../videos/".

    Returns
    -------
        None.
    """

    print(
        "Code execution started at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    file_list_dict = {}

    file_list_dict["file_list_07days"] = np.sort(glob.glob(f"{image_path}*.png"))[
        -number_of_files:
    ]
    print(f"Number of files: {len(file_list_dict['file_list_07days'])}")

    skip_rate_list = [1, 1, 1, 1]
    for i, key in enumerate(list(file_list_dict.keys())):
        # vid_name = f"{gif_path}{key}.{vid_type}"
        vid_name = f"{gif_path}DSCOVR_07days_hourly_averaged.{vid_type}"
        # If the folder does not exist, create it
        Path(gif_path).mkdir(parents=True, exist_ok=True)
        try:
            gif_maker(
                file_list_dict[key],
                vid_name,
                mode="I",
                skip_rate=skip_rate_list[i],
                vid_type=vid_type,
                fps=5,
                duration=0.5,
            )
        except ValueError as e:
            print(e)
            pass
