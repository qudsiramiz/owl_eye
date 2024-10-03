# -*- coding: utf-8 -*-

# Code to read a net cdf file
import datetime
import glob
import os
import time
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from scipy.io import netcdf
from pathlib import Path


def plot_figures_dsco_07days(number_of_days=30):
    """
    Download and plot the data from DSCOVR for the last 07 days starting from the present time.
    """

    print(
        "Code execution for DSCOVR 07 days data started at at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    t_unix = datetime.datetime(1970, 1, 1)
    time_now = (datetime.datetime.utcnow() - t_unix).total_seconds()

    # Set the start and end time (in unix time, 07 days interval)
    time_start = time_now - (60 * 60 * 24 * number_of_days)
    time_end = time_now

    # Change the directory to the data directory
    data_dir = "dscovr_data/"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(data_dir)

    time_mag_data = time_start

    while time_mag_data < time_end:
        year = datetime.datetime.fromtimestamp(time_mag_data).strftime("%Y")
        month = datetime.datetime.fromtimestamp(time_mag_data).strftime("%m")
        day = datetime.datetime.fromtimestamp(time_mag_data).strftime("%d")
        url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
        mag_file_name = (
            f"oe_m1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz"
        )

        # Check if the mag file exists in the directory, if not then download it
        mag_pattern = re.compile(f"{mag_file_name[:-4]}")
        mag_file_list_all = os.listdir(os.getcwd())

        mag_file_exist_list = list(filter(mag_pattern.match, mag_file_list_all))
        if len(mag_file_exist_list) == 0:
            print(
                f"Downloading mag data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            )
            os.system(f"""wget -q -r -np -nc -nH -nd -A {mag_file_name} {url}""")
            time_mag_data += 60 * 60 * 24
        else:
            print(
                f"Mag data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} already exists"
            )
            time_mag_data += 60 * 60 * 24

    time_plasma_data = time_start
    while time_plasma_data < time_end:
        year = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%Y")
        month = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%m")
        day = datetime.datetime.utcfromtimestamp(time_plasma_data).strftime("%d")
        url = f"https://www.ngdc.noaa.gov/dscovr/data/{year}/{str(month).zfill(2)}/"
        plas_file_name = (
            f"oe_f1m_dscovr_s{year}{str(month).zfill(2)}{str(day).zfill(2)}*.gz"
        )

        # Check if the plasma file exists in the directory, if not then download it
        plas_pattern = re.compile(f"{plas_file_name[:-4]}")
        plas_file_list_all = os.listdir(os.getcwd())

        plas_file_exist_list = list(filter(plas_pattern.match, plas_file_list_all))
        if len(plas_file_exist_list) == 0:
            print(
                f"Downloading plasma data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
            )
            os.system(f"""wget -q -r -np -nc -nH -nd -A {plas_file_name} {url}""")
            time_plasma_data += 60 * 60 * 24
        else:
            print(
                f"Plasma data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} already exists"
            )
            time_plasma_data += 60 * 60 * 24

    print("Downloading complete\n")

    # Unzip the files and delete the zipped files
    print("Unzipping the files\n")
    try:
        os.system("gunzip oe_*.gz")
        print("Unzipping complete\n")
    except Exception:
        print("No files to unzip\n")

    os.chdir("/home/vetinari/Desktop/git/owl_eye/codes/")

    plas_file_list = np.sort(glob.glob(f"{data_dir}/oe_f1m_*.nc"))[-number_of_days:]
    mag_file_list = np.sort(glob.glob(f"{data_dir}/oe_m1m_*.nc"))[-number_of_days:]

    print(f"Reading data from {len(mag_file_list)} files\n")

    df_mag_list = [None] * len(mag_file_list)
    df_plas_list = [None] * len(plas_file_list)

    count = 0
    for mag_file, plas_file in zip(mag_file_list, plas_file_list):

        with netcdf.netcdf_file(mag_file, "r", mmap=True) as mag_data:
            mag_data_time = np.copy(mag_data.variables["time"][:])
            mag_data_bx_gsm = np.copy(
                mag_data.variables["bx_gsm"][:].byteswap().newbyteorder()
            )
            mag_data_by_gsm = np.copy(
                mag_data.variables["by_gsm"][:].byteswap().newbyteorder()
            )
            mag_data_bz_gsm = np.copy(
                mag_data.variables["bz_gsm"][:].byteswap().newbyteorder()
            )
            mag_data_bt_gsm = np.copy(
                mag_data.variables["bt"][:].byteswap().newbyteorder()
            )

        with netcdf.netcdf_file(plas_file, "r", mmap=True) as plas_data:
            plas_data_time = np.copy(plas_data.variables["time"][:])
            plas_data_np = np.copy(
                plas_data.variables["proton_density"][:].byteswap().newbyteorder()
            )
            plas_data_vp = np.copy(
                plas_data.variables["proton_speed"][:].byteswap().newbyteorder()
            )
            plas_data_tp = np.copy(
                plas_data.variables["proton_temperature"][:].byteswap().newbyteorder()
            )

        mag_data_time_utc = np.array(
            [datetime.datetime.utcfromtimestamp(t / 1.0e3) for t in mag_data_time]
        )
        plas_data_time_utc = np.array(
            [datetime.datetime.utcfromtimestamp(t / 1.0e3) for t in plas_data_time]
        )

        df_mag_list[count] = pd.DataFrame(
            {
                "bx_gsm": mag_data_bx_gsm,
                "by_gsm": mag_data_by_gsm,
                "bz_gsm": mag_data_bz_gsm,
                "bt": mag_data_bt_gsm,
            },
            index=mag_data_time_utc,
        )
        df_plas_list[count] = pd.DataFrame(
            {"np": plas_data_np, "vp": plas_data_vp, "tp": plas_data_tp},
            index=plas_data_time_utc,
        )

        count += 1

    df_dscovr_mag = pd.concat(df_mag_list, axis=0)
    df_dscovr_plas = pd.concat(df_plas_list, axis=0)

    df_dscovr_mag_plas = pd.concat([df_dscovr_mag, df_dscovr_plas], axis=1)

    # Replace dscovr data gaps with NaN
    df_dscovr_mag_plas.replace([-999.9, -99999.0, -100000], np.nan, inplace=True)

    # Get rid of rows with same time stamp as index while keeping the latest value
    df_dscovr_hc = df_dscovr_mag_plas[~df_dscovr_mag_plas.index.duplicated(keep="last")]

    # Save the flux data to the dataframe
    df_dscovr_hc["flux"] = df_dscovr_hc.np * df_dscovr_hc.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dscovr_hc["bm"] = np.sqrt(
        df_dscovr_hc.bx_gsm**2 + df_dscovr_hc.by_gsm**2 + df_dscovr_hc.bz_gsm**2
    )

    # Compute the IMF clock angle and save it to dataframe
    df_dscovr_hc["theta_c"] = np.arctan2(df_dscovr_hc.by_gsm, df_dscovr_hc.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dscovr_hc["p_dyn"] = 1.6726e-6 * 1.15 * df_dscovr_hc.np * df_dscovr_hc.vp**2

    # Compute 1 hr rolling average for each of the parameters and save it to the dataframe
    df_dscovr = df_dscovr_hc.rolling("1h", center=True).median()
    df_dscovr_all = df_dscovr.copy()

    # Define the plot parameters
    # cmap = plt.cm.viridis
    # pad = 0.02
    # clabelpad = 10
    # labelsize = 22
    ticklabelsize = 20
    # cticklabelsize = 15
    # clabelsize = 15
    ticklength = 12
    tickwidth = 1.5
    # mticklength = 8
    # cticklength = 5
    # mcticklength = 4
    # labelrotation = 0
    xlabelsize = 20
    ylabelsize = 20
    alpha = 0.3
    # bar_color = 'k'

    ms = 2
    lw = 2
    # ncols = 2

    try:
        plt.close("all")
    except Exception:
        pass

    t_start = df_dscovr.index.min()
    t_end = df_dscovr_hc.index[-1]

    while t_end <= df_dscovr_hc.index[-1]:
        print(f"Plotting figures for DSCOVR 07 days starting from {t_start} to {t_end}")
        # df_dscovr = df_dscovr_hc[(df_dscovr_hc.index >= t_start) & (df_dscovr_hc.index <= t_end)]
        df_dscovr = df_dscovr_all.loc[t_start:t_end]

        # Use the dark theme for the plot
        plt.style.use("dark_background")

        tick_color = "w"  # color of the tick lines
        label_color = "w"  # color of the tick labels

        fig = plt.figure(
            num=None,
            figsize=(34.56 / 2, 39.84 / 2),
            dpi=100,
            facecolor="k",
            edgecolor="gray",
        )
        fig.subplots_adjust(
            left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
        )
        fig.suptitle("7 days DSCOVR Real Time Data", fontsize=24)

        # Magnetic field plot (x-component)
        gs = fig.add_gridspec(7, 1)
        axs1 = fig.add_subplot(gs[0:1, 0])
        _ = axs1.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bx_gsm.values,
            "r-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_x$",
        )
        _ = axs1.plot(
            df_dscovr.index.values,
            df_dscovr.bx_gsm.values,
            "r-",
            lw=lw,
            ms=ms,
            label=r"$B_x$",
        )

        # im1d = axs1.plot(df_dscovr.index, df_dscovr.bm, 'k-.', lw=lw, ms=ms, label=r'$|\vec{B}|$')
        # im1e = axs1.plot(df_dscovr.index, -df_dscovr.bm, 'k-.', lw=lw, ms=ms)

        if df_dscovr.bm.isnull().all():
            axs1.set_ylim([-1, 1])
        else:
            axs1.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.bx_gsm),
                1.1 * np.nanmax(df_dscovr_all.bx_gsm),
            )

        axs1.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs1.set_ylabel(r"$B_{\rm x}$ (GSM) [nT]", fontsize=20)

        # Magnetic field plot (y-component)
        axs2 = fig.add_subplot(gs[1:2, 0])
        _ = axs2.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.by_gsm.values,
            "c-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_y$",
        )
        _ = axs2.plot(
            df_dscovr.index.values,
            df_dscovr.by_gsm.values,
            "c-",
            lw=lw,
            ms=ms,
            label=r"$B_y$",
        )

        if df_dscovr.bm.isnull().all():
            axs2.set_ylim([-1, 1])
        else:
            axs2.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.by_gsm),
                1.1 * np.nanmax(df_dscovr_all.by_gsm),
            )

        axs2.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs2.set_ylabel(r"$B_{\rm y}$ (GSM) [nT]", fontsize=20)

        # Magnetic field plot (z-component)
        axs3 = fig.add_subplot(gs[2:3, 0])
        _ = axs3.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bz_gsm.values,
            color="bisque",
            ls="-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"$B_z$",
        )
        _ = axs3.plot(
            df_dscovr.index.values,
            df_dscovr.bz_gsm.values,
            color="bisque",
            lw=lw,
            ms=ms,
            label=r"$B_z$",
        )

        if df_dscovr.bm.isnull().all():
            axs3.set_ylim([-1, 1])
        else:
            axs3.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.bz_gsm),
                1.1 * np.nanmax(df_dscovr_all.bz_gsm),
            )

        axs3.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs3.set_ylabel(r"$B_{\rm z}$ (GSM) [nT]", fontsize=20)

        # Magnetic field plot (z-component)
        axs4 = fig.add_subplot(gs[3:4, 0])
        _ = axs4.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.bm.values,
            "w-",
            alpha=0.3,
            lw=lw,
            ms=ms,
            label=r"|$\vec{B}$|",
        )
        _ = axs4.plot(
            df_dscovr.index.values,
            df_dscovr.bm.values,
            "w-",
            lw=lw,
            ms=ms,
            label=r"$B_z$",
        )

        if df_dscovr.bm.isnull().all():
            axs4.set_ylim([-1, 1])
        else:
            axs4.set_ylim(
                0.8 * np.nanmin(df_dscovr_all.bm), 1.2 * np.nanmax(df_dscovr_all.bm)
            )

        axs4.set_xlim(df_dscovr.index.min(), df_dscovr.index.max())
        axs4.set_ylabel(r"B [nT]", fontsize=20, color=label_color)

        axs4.set_yscale("log")

        # Density plot
        axs5 = fig.add_subplot(gs[4:5, 0], sharex=axs1)
        _ = axs5.plot(
            df_dscovr.index.values,
            df_dscovr.np.values,
            color="fuchsia",
            ls="-",
            lw=lw,
            ms=ms,
            label=r"$n_p$",
        )
        axs5.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.np.values,
            color="fuchsia",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.np.isnull().all():
            axs5.set_ylim([0, 1])
        else:
            axs5.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.np), 1.1 * np.nanmax(df_dscovr_all.np)
            )

        axs5.set_yscale("log")
        axs5.set_ylabel(r"$n_{\rm p} [1/\rm{cm^{3}}]$", fontsize=ylabelsize)

        axs5a = axs5.twinx()
        _ = axs5a.plot(
            df_dscovr.index.values,
            df_dscovr.flux.values,
            color="oldlace",
            ls="-",
            lw=lw,
            ms=ms,
            label=r"flux",
        )
        axs5a.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.flux.values,
            color="oldlace",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.flux.isnull().all():
            axs5a.set_ylim([0, 1])
        else:
            axs5a.set_ylim(
                np.nanmin([0.9 * np.nanmin(df_dscovr_all.flux), 2.4]),
                np.nanmax([1.1 * np.nanmax(df_dscovr_all.flux), 3.3]),
            )

        axs5a.set_ylabel(
            r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize
        )
        axs5a.set_yscale("log")
        # Speed plot
        axs6 = fig.add_subplot(gs[5:6, 0], sharex=axs1)
        _ = axs6.plot(
            df_dscovr.index.values,
            df_dscovr.vp.values,
            color="ivory",
            ls="-",
            lw=lw,
            ms=ms,
            label=r"$V_p$",
        )
        axs6.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.vp.values,
            color="ivory",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.vp.isnull().all():
            axs6.set_ylim([0, 1])
        else:
            axs6.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.vp), 1.1 * np.nanmax(df_dscovr_all.vp)
            )

        axs6.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize)

        # Flux plot
        axs6a = axs6.twinx()
        _ = axs6a.plot(
            df_dscovr.index.values,
            df_dscovr.tp.values / 1.0e5,
            color="lime",
            ls="-",
            lw=lw,
            ms=ms,
            label=r"flux",
        )
        axs6a.plot(
            df_dscovr_hc.index.values,
            df_dscovr_hc.tp.values,
            color="lime",
            lw=1,
            alpha=alpha,
        )

        if df_dscovr.tp.isnull().all():
            axs6a.set_ylim([0, 1])
        else:
            axs6a.set_ylim(
                0.9 * np.nanmin(df_dscovr_all.tp) / 1.0e5,
                1.1 * np.nanmax(df_dscovr_all.tp) / 1.0e5,
            )

        axs6a.set_yscale("log")

        axs6a.set_ylabel(r"$T_{\rm p} [10^5~\rm{K}]$", fontsize=ylabelsize)

        # Set axis tick-parameters
        axs1.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="r",
        )
        axs1.spines["left"].set_color("r")
        axs1.spines["top"].set_color("w")
        axs1.spines["left"].set_linewidth(2)
        axs1.yaxis.label.set_color("r")
        axs1.yaxis.set_label_position("left")

        axs2.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="c",
        )
        axs2.spines["right"].set_color("c")
        axs2.spines["right"].set_linewidth(2)
        axs2.yaxis.label.set_color("c")
        axs2.yaxis.set_label_position("right")

        axs3.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=True,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="bisque",
        )
        axs3.spines["left"].set_color("bisque")
        axs3.spines["left"].set_linewidth(2)
        axs3.yaxis.label.set_color("bisque")
        axs3.yaxis.set_label_position("left")

        axs4.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors=tick_color,
        )
        axs4.spines["left"].set_color("w")
        axs4.spines["left"].set_linewidth(2)
        axs4.yaxis.label.set_color(label_color)
        axs4.yaxis.set_label_position("right")

        axs5.tick_params(
            which="both",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="fuchsia",
        )
        axs5.spines["right"].set_color("fuchsia")
        axs5.spines["right"].set_linewidth(2)
        axs5.yaxis.label.set_color("fuchsia")
        axs5.yaxis.set_label_position("right")

        axs5a.tick_params(
            which="both",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="oldlace",
        )
        axs5a.spines["left"].set_color("oldlace")
        axs5a.spines["left"].set_linewidth(2)
        axs5a.yaxis.label.set_color("oldlace")
        axs5a.yaxis.set_label_position("left")

        axs6.tick_params(
            which="both",
            axis="y",
            direction="in",
            left=True,
            labelleft=True,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=False,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="ivory",
        )
        axs6.spines["left"].set_color("ivory")
        axs6.spines["left"].set_linewidth(2)
        axs6.yaxis.label.set_color("ivory")
        axs6.yaxis.set_label_position("left")

        axs6a.tick_params(
            which="both",
            axis="y",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=True,
            labelright=True,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors="lime",
        )
        axs6a.spines["right"].set_color("lime")
        axs6a.spines["right"].set_linewidth(2)
        axs6a.yaxis.label.set_color("lime")
        axs6a.yaxis.set_label_position("right")

        axs6.tick_params(
            which="both",
            axis="x",
            direction="in",
            left=False,
            labelleft=False,
            top=True,
            labeltop=False,
            right=False,
            labelright=False,
            bottom=True,
            labelbottom=True,
            width=tickwidth,
            length=ticklength,
            labelsize=ticklabelsize,
            labelrotation=0,
            colors=tick_color,
        )
        date_form = DateFormatter("%m-%d")
        # axs6.xaxis.label.set_color('k')
        axs6.set_xlabel(
            f"Date and Time starting on {int(df_dscovr.index.year[count])}- \
            {str(int(df_dscovr.index.month[count])).zfill(2)}- \
            {str(int(df_dscovr.index.day[count])).zfill(2)} (UTC) [MM-DD]",
            fontsize=xlabelsize,
            color=label_color,
        )
        axs6.xaxis.set_major_formatter(date_form)

        _ = f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

        # Properly define the folder and figure name
        folder_name = "~/Dropbox/rt_sw_oe/"
        folder_name = Path(folder_name).expanduser()
        # cd into the folder using Path
        Path(folder_name).mkdir(parents=True, exist_ok=True)

        fig_name = f"rt_sw_dsco_parameters_{number_of_days}day.png"

        fig_name = folder_name / fig_name
        print(f"Figure saved at: {fig_name}")
        plt.savefig(
            fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
        )

        plt.close("all")
        print(f"Figure saved for DSCOVR 07 days starting from {t_start} to {t_end}\n")

        t_start = t_start + datetime.timedelta(hours=1)
        t_end = t_end + datetime.timedelta(hours=1)
    return df_dscovr_hc


if __name__ == "__main__":
    df_dscovr_hc = plot_figures_dsco_07days(number_of_days=7)
    df_dscovr_hc = plot_figures_dsco_07days(number_of_days=54)
