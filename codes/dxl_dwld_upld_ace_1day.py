#!/home/cephadrius/.cache/pypoetry/virtualenvs/dxl-bu-jVBu7xQf-py3.10/bin/python
# -*- coding: utf-8 -*-
import datetime
import sched
import time

import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

s = sched.scheduler(time.time, time.sleep)


def plot_figures_ace_1day():
    # for foo in range(1):
    """
    Download and upload data the ACE database hosted at https://services.swpc.noaa.gov/text
    """
    # Set up the time to run the job
    # s.enter(60, 1, plot_figures_ace_1day, (sc,))

    # start = time.time()
    print(
        f"Code execution for ace 1day data started at at (UTC):"
        + f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # URL of ace files
    ace_url_mag = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    ace_url_plas = (
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    )
    ace_url_eph = "https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json"

    ace_key_list_mag = [
        "time_tag",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "lon_gsm",
        "lat_gsm",
        "bt",
    ]
    ace_key_list_plas = ["time_tag", "np", "vp", "Tp"]
    ace_key_list_eph = [
        "time_tag",
        "x_gse",
        "y_gse",
        "z_gse",
        "vx_gse",
        "vy_gse",
        "vz_gse",
        "x_gsm",
        "y_gsm",
        "z_gsm",
        "vx_gsm",
        "vy_gsm",
        "vz_gsm",
    ]

    df_ace_mag = pd.read_json(ace_url_mag, orient="columns")
    df_ace_plas = pd.read_json(ace_url_plas, orient="columns")
    df_ace_eph = pd.read_json(ace_url_eph, orient="columns")

    # Drop the first row of the dataframe to get rid of all strings
    df_ace_mag.drop([0], inplace=True)
    df_ace_plas.drop([0], inplace=True)
    df_ace_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_ace_mag.columns = ace_key_list_mag
    df_ace_plas.columns = ace_key_list_plas
    df_ace_eph.columns = ace_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_ace_mag.index = pd.to_datetime(df_ace_mag.time_tag)
    df_ace_plas.index = pd.to_datetime(df_ace_plas.time_tag)
    df_ace_eph.index = pd.to_datetime(df_ace_eph.time_tag)

    # Drop the time_tag column
    df_ace_mag.drop(["time_tag"], axis=1, inplace=True)
    df_ace_plas.drop(["time_tag"], axis=1, inplace=True)
    df_ace_eph.drop(["time_tag"], axis=1, inplace=True)

    df_ace_eph = df_ace_eph[
        (
            df_ace_eph.index
            >= np.nanmin([df_ace_mag.index.min(), df_ace_plas.index.min()])
        )
        & (
            df_ace_eph.index
            <= np.nanmax([df_ace_mag.index.max(), df_ace_plas.index.max()])
        )
    ]

    df_ace = pd.concat([df_ace_mag, df_ace_plas, df_ace_eph], axis=1)

    # for key in df_ace.keys():
    #     df_ace[key] = pd.to_numeric(df_ace[key])
    df_ace = df_ace.apply(pd.to_numeric)
    # Save the flux data to the dataframe
    df_ace["flux"] = df_ace.np * df_ace.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_ace["bm"] = np.sqrt(df_ace.bx_gsm**2 + df_ace.by_gsm**2 + df_ace.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_ace["theta_c"] = np.arctan2(df_ace.by_gsm, df_ace.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_ace["p_dyn"] = 1.6726e-6 * 1.15 * df_ace.np * df_ace.vp**2

    # Find index in dataframe which is closest to the value of solar wind assuming 35 minutes
    # propagation time
    indx_min = min(
        range(len(df_ace.index)),
        key=lambda i: abs(
            df_ace.index[i] - df_ace.index[-1] + datetime.timedelta(minutes=35)
        ),
    )

    # Set a new series corresponding to the index for the closest value of solar wind
    df_param = df_ace.iloc[indx_min]

    # Define the parameter for computing the total magnetic field from Tsyganenko model (T04)
    # NOTE: For the Tsyganenko model, the elemnts of 'param' are solar wind dynamic pressure, DST,
    # y-component of IMF, z-component of IMF, and 6 zeros.
    param = [df_param.p_dyn, 1, df_param.by_gsm, df_param.bz_gsm, 0, 0, 0, 0, 0, 0]

    # Compute the unix time and the dipole tilt angle
    t_unix = datetime.datetime(1970, 1, 1)
    time_dipole = (df_ace.index[indx_min] - t_unix).total_seconds()
    ps = gp.recalc(time_dipole)

    # Compute the dipole tilt angle correction, indegrees,  to be applied to the cusp locations
    # NOTE: The correctionvalue computed  here is based on the values given in Newell et al. (2006),
    # doi:10.1029/2006JA011731, 2006
    dipole_correction = -0.046 * ps * 180 / np.pi

    # Compute the location of cusp based on different coupling equations
    df_ace["lambda_phi"] = (
        -3.65e-2
        * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4) ** (2 / 3)
        + 77.2
        + dipole_correction
    )
    df_ace["lambda_wav"] = (
        -2.27e-3 * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4)
        + 78.5
        + dipole_correction
    )
    df_ace["lambda_vas"] = (
        -2.14e-4
        * df_ace.p_dyn ** (1 / 6)
        * df_ace.vp ** (4 / 3)
        * df_ace.bt
        * np.sin(df_ace.theta_c) ** 4
        + 78.3
        + dipole_correction
    )
    df_ace["lambda_ekl"] = (
        -1.90e-3 * df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 2
        + 78.9
        + dipole_correction
    )

    # Make a copy of the dataframe at original cadence
    df_ace_hc = df_ace.copy()

    # Compute 1 hour rolling average for each of the parameters and save it to the dataframe
    df_ace = df_ace.rolling("1H", center=True).median()
    # Define the plot parameters
    # cmap = plt.cm.viridis
    # pad = 0.02
    # clabelpad = 10
    labelsize = 22
    ticklabelsize = 20
    # cticklabelsize = 15
    # clabelsize = 15
    ticklength = 6
    tickwidth = 1.0
    # mticklength = 4
    # cticklength = 5
    # mcticklength = 4
    # labelrotation = 0
    xlabelsize = 20
    ylabelsize = 20
    alpha = 0.3
    bar_color = "k"

    ms = 2
    lw = 2
    ncols = 2
    alpha = 0.3

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_ace.index.max() - datetime.timedelta(minutes=30)
    t2 = df_ace.index.max() - datetime.timedelta(minutes=40)

    fig = plt.figure(
        num=None, figsize=(10, 13), dpi=200, facecolor="w", edgecolor="gray"
    )
    fig.subplots_adjust(
        left=0.01, right=0.95, top=0.95, bottom=0.01, wspace=0.02, hspace=0.0
    )

    # Magnetic field plot
    gs = fig.add_gridspec(6, 1)
    axs1 = fig.add_subplot(gs[0, 0])
    axs1.plot(
        df_ace.index.values, df_ace.bx_gsm.values, "r-", lw=lw, ms=ms, label=r"$B_x$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.by_gsm.values, "b-", lw=lw, ms=ms, label=r"$B_y$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.bz_gsm.values, "g-", lw=lw, ms=ms, label=r"$B_z$"
    )
    axs1.plot(
        df_ace.index.values, df_ace.bm.values, "k-.", lw=lw, ms=ms, label=r"$|\vec{B}|$"
    )
    axs1.plot(df_ace.index.values, -df_ace.bm.values, "k-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_ace.bm), 1.1 * np.nanmax(df_ace.bm))

    axs1.set_xlim(df_ace.index.min(), df_ace.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
    lgnd1 = axs1.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd1.legendHandles[0]._sizes = [labelsize]
    fig.suptitle(f"1 Day ace Real Time Data", fontsize=22)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    axs2.plot(df_ace.index.values, df_ace.np.values, "r-", lw=lw, ms=ms, label=r"$n_p$")
    axs2.plot(df_ace_hc.index.values, df_ace_hc.np.values, color="r", lw=1, alpha=alpha)
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.np.isnull().all():
        axs2.set_ylim([-1, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_ace.np), 1.1 * np.nanmax(df_ace.np))

    lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize)

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    axs3.plot(df_ace.index.values, df_ace.vp.values, "b-", lw=lw, ms=ms, label=r"$V_p$")
    axs3.plot(df_ace_hc.index.values, df_ace_hc.vp.values, color="b", lw=1, alpha=alpha)
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.vp.isnull().all():
        axs3.set_ylim([-1, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_ace.vp), 1.1 * np.nanmax(df_ace.vp))

    lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd3.legendHandles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize)

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    axs4.plot(
        df_ace.index.values, df_ace.flux.values, "g-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.plot(
        df_ace_hc.index.values, df_ace_hc.flux.values, color="g", lw=1, alpha=alpha
    )
    im4a = axs4.axhline(
        y=2.9, xmin=0, xmax=1, color="r", ls="-", lw=lw, ms=ms, label=r"cut-off"
    )
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.flux.isnull().all():
        axs4.set_ylim([-1, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_ace.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_ace.flux), 3.3]),
        )

    lgnd4 = axs4.legend(fontsize=labelsize, loc="best", ncol=ncols)
    lgnd4.legendHandles[0]._sizes = [labelsize]
    axs4.set_ylabel(r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize)

    # Cusp latitude plot
    axs5 = fig.add_subplot(gs[4:, 0], sharex=axs1)

    min_lambda = np.nanmin(
        [
            np.nanmin(df_ace.lambda_phi),
            np.nanmin(df_ace.lambda_wav),
            np.nanmin(df_ace.lambda_vas),
            np.nanmin(df_ace.lambda_ekl),
        ]
    )
    max_lambda = np.nanmax(
        [
            np.nanmax(df_ace.lambda_phi),
            np.nanmax(df_ace.lambda_wav),
            np.nanmax(df_ace.lambda_vas),
            np.nanmax(df_ace.lambda_ekl),
        ]
    )

    axs5.plot(
        df_ace.index.values,
        df_ace.lambda_phi.values,
        "r-",
        lw=lw,
        ms=ms,
        label=r"$d\phi/dt$",
    )
    axs5.plot(
        df_ace.index.values, df_ace.lambda_wav.values, "b-", lw=lw, ms=ms, label=r"WAV"
    )
    axs5.plot(
        df_ace.index.values, df_ace.lambda_vas.values, "g-", lw=lw, ms=ms, label=r"Vas"
    )
    axs5.plot(
        df_ace.index.values, df_ace.lambda_ekl.values, "m-", lw=lw, ms=ms, label=r"EKL"
    )
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if (
        df_ace.lambda_phi.isnull().all()
        and df_ace.lambda_wav.isnull().all()
        and df_ace.lambda_vas.isnull().all()
        and df_ace.lambda_ekl.isnull().all()
    ):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_lambda, 1.03 * max_lambda)

    lgnd5 = axs5.legend(fontsize=labelsize, loc="best", ncol=4)
    lgnd5.legendHandles[0]._sizes = [labelsize]

    axs5.set_xlabel(f"Time on {df_ace.index.date[0]} [UTC]", fontsize=xlabelsize)
    axs5.set_ylabel(r"$\lambda[^\circ]$", fontsize=ylabelsize)

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
    )

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
    )
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
    )

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
    )
    axs4.yaxis.set_label_position("right")

    axs5.tick_params(
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        bottom=True,
        labelbottom=True,
        width=tickwidth,
        length=ticklength,
        labelsize=ticklabelsize,
        labelrotation=0,
    )
    axs5.yaxis.set_label_position("left")

    date_form = DateFormatter("%H-%M")
    axs5.xaxis.set_major_formatter(date_form)

    figure_time = f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"

    axs3.text(
        -0.1,
        0.5,
        f"Figure plotted on {figure_time[0:10]} at {figure_time[11:]} UTC",
        ha="right",
        va="center",
        transform=axs3.transAxes,
        fontsize=20,
        rotation="vertical",
    )

    fig_name = f"/home/cephadrius/Dropbox/DXL-Figure/sw_ace_parameters_1day.png"
    fig_name_git = "../figures/sw_ace_parameters_1day.png"
    fig_name_gdr = f"/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/sw_ace_parameters_1day.png"

    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300)
    plt.savefig(
        fig_name_git, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
    )
    plt.savefig(
        fig_name_gdr, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
    )

    axs1.set_ylim([-22, 22])
    axs2.set_ylim([0, 40])
    axs3.set_ylim([250, 700])
    axs4.set_ylim([0, 20])
    axs5.set_ylim([60, 85])

    t = int(datetime.datetime.today().replace(tzinfo=datetime.timezone.utc).timestamp())
    fig_name_hist = f"/mnt/cephadrius/bu_research/dxl/figures/historical/ace/1day/sw_ace_parameters_1day_{t}.png"
    plt.savefig(
        fig_name_hist, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300
    )
    # plt.tight_layout()
    # plt.close()
    print(
        "Figure saved at (UTC):"
        + f"{datetime.datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # print(f'It took {round(time.time() - start, 3)} seconds')
    # return df


# s.enter(0, 1, plot_figures_ace_1day, (s,))
# s.run()

if __name__ == "__main__":
    plot_figures_ace_1day()
