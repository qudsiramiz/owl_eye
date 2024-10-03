# -*- coding: utf-8 -*-
import datetime
import time
import geopack.geopack as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter
from pathlib import Path

# Set the dark mode for the plots
plt.style.use("dark_background")


def plot_figures_dsco_1day(sc=None):
    # for foo in range(1):
    """
    Download and upload data the dsco database hosted at
    https://services.swpc.noaa.gov/text/dsco-swepam-1-day.json
    The data is then processed to compute the solar wind parameters and the magnetopause radius using
    the Shue et al., 1998 model, the Yang et al., 2011 model and the Lin et al., 2008 model.
    The data is then plotted and saved to a file in the Dropbox folder. The function is scheduled to
    run at regular intervals using the sched module in Python standard library to update the plots at
    regular intervals of time.

    Parameters
    ----------
    sc : sched.scheduler
        The scheduler object

    Returns
    -------
    df_dsco_hc : pandas.DataFrame
        The dataframe containing the solar wind parameters

    """
    # Set up the time to run the job
    # s.enter(0, 1, m_codes.update_progress_bar, (sc, 0, 52))
    # s.enter(60, 1, plot_figures_dsco_1day, (sc,))

    # start = time.time()
    print(
        "\nCode execution for DSCOVR 1day data started at at (UTC):"
        + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # URL of dscovr files
    dscovr_url_mag = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    dscovr_url_plas = (
        "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    )
    dscovr_url_eph = (
        "https://services.swpc.noaa.gov/products/solar-wind/ephemerides.json"
    )

    dscovr_key_list_mag = [
        "time_tag",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "lon_gsm",
        "lat_gsm",
        "bt",
    ]
    dscovr_key_list_plas = ["time_tag", "np", "vp", "Tp"]
    dscovr_key_list_eph = [
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

    df_dsco_mag = pd.read_json(dscovr_url_mag, orient="columns")
    df_dsco_plas = pd.read_json(dscovr_url_plas, orient="columns")
    df_dsco_eph = pd.read_json(dscovr_url_eph, orient="columns")

    # Drop the first row of the dataframe to get rid of all strings
    df_dsco_mag.drop([0], inplace=True)
    df_dsco_plas.drop([0], inplace=True)
    df_dsco_eph.drop([0], inplace=True)

    # Set column names to the list of keys
    df_dsco_mag.columns = dscovr_key_list_mag
    df_dsco_plas.columns = dscovr_key_list_plas
    df_dsco_eph.columns = dscovr_key_list_eph

    # Set the index to the time_tag column and convert it to a datetime object
    df_dsco_mag.index = pd.to_datetime(df_dsco_mag.time_tag)
    df_dsco_plas.index = pd.to_datetime(df_dsco_plas.time_tag)
    df_dsco_eph.index = pd.to_datetime(df_dsco_eph.time_tag)

    # Drop the time_tag column
    df_dsco_mag.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_plas.drop(["time_tag"], axis=1, inplace=True)
    df_dsco_eph.drop(["time_tag"], axis=1, inplace=True)

    df_dsco_eph = df_dsco_eph[
        (
            df_dsco_eph.index
            >= np.nanmin([df_dsco_mag.index.min(), df_dsco_plas.index.min()])
        )
        & (
            df_dsco_eph.index
            <= np.nanmax([df_dsco_mag.index.max(), df_dsco_plas.index.max()])
        )
    ]

    df_dsco = pd.concat([df_dsco_mag, df_dsco_plas, df_dsco_eph], axis=1)

    # for key in df_dsco.keys():
    #     df_dsco[key] = pd.to_numeric(df_dsco[key])
    df_dsco = df_dsco.apply(pd.to_numeric)
    # Save the flux data to the dataframe
    df_dsco["flux"] = df_dsco.np * df_dsco.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_dsco["bm"] = np.sqrt(df_dsco.bx_gsm**2 + df_dsco.by_gsm**2 + df_dsco.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_dsco["theta_c"] = np.arctan2(df_dsco.by_gsm, df_dsco.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_dsco["p_dyn"] = 1.6726e-6 * 1.15 * df_dsco.np * df_dsco.vp**2

    # Get the unix time for all the time tags
    df_dsco["unix_time"] = df_dsco.index.astype(int) // 10**9

    # Compute the dipole tilt angle
    for i in range(len(df_dsco)):
        tilt_angle_gp = gp.recalc(df_dsco.unix_time.iloc[i])
        df_dsco.loc[df_dsco.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    # Compute the dipole tilt angle correction, indegrees,  to be applied to the cusp locations
    # NOTE: The correctionvalue computed  here is based on the values given in Newell et al. (2006),
    # doi:10.1029/2006JA011731, 2006
    df_dsco["dipole_tilt_corr"] = -0.046 * df_dsco.dipole_tilt * 180 / np.pi

    # Compute the location of cusp based on different coupling equations
    df_dsco["lambda_phi"] = (
        -3.65e-2
        * (df_dsco.vp * df_dsco.bt * np.sin(df_dsco.theta_c / 2.0) ** 4) ** (2 / 3)
        + 77.2
        + df_dsco.dipole_tilt_corr
    )

    df_dsco["lambda_wav"] = (
        -2.27e-3 * (df_dsco.vp * df_dsco.bt * np.sin(df_dsco.theta_c / 2.0) ** 4)
        + 78.5
        + df_dsco.dipole_tilt_corr
    )
    df_dsco["lambda_vas"] = (
        -2.14e-4
        * df_dsco.p_dyn ** (1 / 6)
        * df_dsco.vp ** (4 / 3)
        * df_dsco.bt
        * np.sin(df_dsco.theta_c) ** 4
        + 78.3
        + df_dsco.dipole_tilt_corr
    )
    df_dsco["lambda_ekl"] = (
        -1.90e-3 * df_dsco.vp * df_dsco.bt * np.sin(df_dsco.theta_c / 2.0) ** 2
        + 78.9
        + df_dsco.dipole_tilt_corr
    )

    # Make a copy of the dataframe at original cadence
    df_dsco_hc = df_dsco.copy()

    # Compute 1 hour rolling average for each of the parameters and save it to the dataframe
    df_dsco = df_dsco.rolling("h", center=True).median()
    # Define the plot parameters
    # cmap = plt.cm.viridis
    # pad = 0.02
    # clabelpad = 10
    # labelsize = 22
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
    bar_color = "turquoise"

    ms = 2
    lw = 2
    # ncols = 2
    alpha = 0.3

    try:
        plt.close("all")
    except Exception:
        pass

    t1 = df_dsco.index.max() - datetime.timedelta(minutes=30)
    t2 = df_dsco.index.max() - datetime.timedelta(minutes=40)

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
        df_dsco.index.values, df_dsco.bx_gsm.values, "r-", lw=lw, ms=ms, label=r"$B_x$"
    )
    axs1.plot(
        df_dsco.index.values, df_dsco.by_gsm.values, "b-", lw=lw, ms=ms, label=r"$B_y$"
    )
    axs1.plot(
        df_dsco.index.values, df_dsco.bz_gsm.values, "g-", lw=lw, ms=ms, label=r"$B_z$"
    )
    axs1.plot(
        df_dsco.index.values,
        df_dsco.bm.values,
        "w-.",
        lw=lw,
        ms=ms,
        label=r"$|\vec{B}|$",
    )
    axs1.plot(df_dsco.index.values, -df_dsco.bm.values, "w-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_dsco.bm), 1.1 * np.nanmax(df_dsco.bm))

    axs1.set_xlim(df_dsco.index.min(), df_dsco.index.max())
    axs1.set_ylabel(r"B [nT]", fontsize=20)
    # lgnd1 = axs1.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd1.legendHandles[0]._sizes = [labelsize]
    # Add a text in the plot right outside the plot along the right edge in the middle
    y_labels = [r"$|\vec{B}|$", r"$B_x$", r"$B_y$", r"$B_z$"]
    y_label_colors = ["w", "r", "b", "g"]
    for i, txt in enumerate(y_labels):
        axs1.text(
            1.01,
            -0.05 + 0.20 * (i + 1),
            txt,
            ha="left",
            va="center",
            transform=axs1.transAxes,
            fontsize=20,
            color=y_label_colors[i],
        )

    fig.suptitle("1 Day DSCOVR Real Time Data", fontsize=22)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    axs2.plot(
        df_dsco.index.values,
        df_dsco.np.values,
        color="bisque",
        ls="-",
        lw=lw,
        ms=ms,
        label=r"$n_p$",
    )
    axs2.plot(
        df_dsco_hc.index.values, df_dsco_hc.np.values, color="bisque", lw=1, alpha=alpha
    )
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.np.isnull().all():
        axs2.set_ylim([-1, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_dsco.np), 1.1 * np.nanmax(df_dsco.np))

    # lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize, color="bisque")

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    axs3.plot(
        df_dsco.index.values, df_dsco.vp.values, "c-", lw=lw, ms=ms, label=r"$V_p$"
    )
    axs3.plot(
        df_dsco_hc.index.values, df_dsco_hc.vp.values, color="c", lw=1, alpha=alpha
    )
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.vp.isnull().all():
        axs3.set_ylim([-1, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_dsco.vp), 1.1 * np.nanmax(df_dsco.vp))

    # lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd3.legend_handles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize, color="c")

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    axs4.plot(
        df_dsco.index.values, df_dsco.flux.values, "w-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.plot(
        df_dsco_hc.index.values, df_dsco_hc.flux.values, color="w", lw=1, alpha=alpha
    )
    axs4.axhline(
        y=2.9, xmin=0, xmax=1, color="r", ls="-", lw=lw, ms=ms, label=r"cut-off"
    )
    axs4.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_dsco.flux.isnull().all():
        axs4.set_ylim([-1, 1])
    else:
        axs4.set_ylim(
            np.nanmin([0.9 * np.nanmin(df_dsco.flux), 2.4]),
            np.nanmax([1.1 * np.nanmax(df_dsco.flux), 3.3]),
        )

    # lgnd4 = axs4.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd4.legend_handles[0]._sizes = [labelsize]
    axs4.set_ylabel(
        r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize, color="w"
    )

    # Cusp latitude plot

    axs5 = fig.add_subplot(gs[4:, 0], sharex=axs1)

    min_lambda = np.nanmin(
        [
            np.nanmin(df_dsco.lambda_phi),
            np.nanmin(df_dsco.lambda_wav),
            np.nanmin(df_dsco.lambda_vas),
            np.nanmin(df_dsco.lambda_ekl),
        ]
    )
    max_lambda = np.nanmax(
        [
            np.nanmax(df_dsco.lambda_phi),
            np.nanmax(df_dsco.lambda_wav),
            np.nanmax(df_dsco.lambda_vas),
            np.nanmax(df_dsco.lambda_ekl),
        ]
    )

    axs5.plot(
        df_dsco.index.values,
        df_dsco.lambda_phi.values,
        "r-",
        lw=lw,
        ms=ms,
    )
    axs5.plot(df_dsco.index.values, df_dsco.lambda_wav.values, "b-", lw=lw, ms=ms)
    axs5.plot(df_dsco.index.values, df_dsco.lambda_vas.values, "g-", lw=lw, ms=ms)
    axs5.plot(df_dsco.index.values, df_dsco.lambda_ekl.values, "m-", lw=lw, ms=ms)
    axs5.axvspan(t1, t2, alpha=alpha, color=bar_color)
    if (
        df_dsco.lambda_phi.isnull().all()
        and df_dsco.lambda_wav.isnull().all()
        and df_dsco.lambda_vas.isnull().all()
        and df_dsco.lambda_ekl.isnull().all()
    ):
        axs5.set_ylim([-1, 1])
    else:
        axs5.set_ylim(0.97 * min_lambda, 1.03 * max_lambda)

    # lgnd5 = axs5.legend(fontsize=labelsize, loc="best", ncol=4)
    # lgnd5.legendHandles[0]._sizes = [labelsize]
    # Add a text in the plot right outside the plot along the right edge in the middle for the y-axis
    y_labels = [r"$d\phi/dt$", r"WAV", r"Vas", r"EKL"]
    y_label_colors = ["r", "b", "g", "m"]
    for i, txt in enumerate(y_labels):
        axs5.text(
            1.01,
            -0.05 + 0.20 * (i + 1),
            txt,
            ha="left",
            va="center",
            transform=axs5.transAxes,
            fontsize=20,
            color=y_label_colors[i],
        )

    axs5.set_xlabel(f"Time on {df_dsco.index.date[0]} [UTC]", fontsize=xlabelsize)
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

    date_form = DateFormatter("%H:%M")
    axs5.xaxis.set_major_formatter(date_form)

    figure_time = (
        f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

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

    # Properly define the folder and figure name
    folder_name = "~/Dropbox/rt_sw_oe/"
    folder_name = Path(folder_name).expanduser()
    # cd into the folder using Path
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    fig_name = "rt_sw_dsco_parameters_1day.png"

    fig_name = folder_name / fig_name
    print(f"Figure saved at: {fig_name}")
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05, format="png", dpi=300)

    # axs1.set_ylim([-22, 22])
    # axs2.set_ylim([0, 40])
    # axs3.set_ylim([250, 700])
    # axs4.set_ylim([0, 20])
    # axs5.set_ylim([60, 85])

    # plt.tight_layout()
    plt.close("all")
    print(
        "Figure saved at (UTC):"
        + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # print(f'It took {round(time.time() - start, 3)} seconds')
    # return df
    return df_dsco_hc


# s.enter(0, 1, plot_figures_dsco_1day, (s,))
# s.run()

# Print that the code has finished running and is waiting for the next update in 60 seconds
print(
    "Code execution for dsco 1day data finished at (UTC):"
    + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
)
# Display a progress bar for the next update


# s.enter(0, 1, plot_figures_dsco_1day, (s,))
# s.run()

if __name__ == "__main__":
    df_dsco_hc = plot_figures_dsco_1day()
