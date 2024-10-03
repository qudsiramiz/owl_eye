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


def plot_figures_ace():
    # for xxx in range(1):
    """
    Download and upload data the ACE database hosted at https://services.swpc.noaa.gov/text
    """
    # Set up the time to run the job
    # s.enter(60, 1, plot_figures_ace, (sc,))

    print(
        "Code execution for ACE 2Hr started at (UTC):"
        + f"{datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

    # Set the font style to Times New Roman
    font = {"family": "serif", "weight": "normal", "size": 10}
    plt.rc("font", **font)
    plt.rc("text", usetex=True)

    # URL of sweap and magnetometer files
    ace_url_mag = "https://services.swpc.noaa.gov/text/ace-magnetometer.txt"
    ace_url_swp = "https://services.swpc.noaa.gov/text/ace-swepam.txt"

    # List of keys for the two files
    ace_key_list_mag = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "bx_gsm",
        "by_gsm",
        "bz_gsm",
        "bt",
        "lat_gsm",
        "lon_gsm",
    ]
    ace_key_list_swp = [
        "year",
        "month",
        "date",
        "utctime",
        "julian_day",
        "doy",
        "s",
        "np",
        "vp",
        "Tp",
    ]

    # Read data from sweap and magnetometer in a dataframe
    df_ace_mag = pd.read_csv(
        ace_url_mag,
        sep=r"\s{1,}",
        skiprows=20,
        names=ace_key_list_mag,
        engine="python",
        dtype={"month": "string", "date": "string", "utctime": "string"},
    )
    df_ace_swp = pd.read_csv(
        ace_url_swp,
        sep=r"\s{1,}",
        skiprows=18,
        names=ace_key_list_swp,
        engine="python",
        dtype={"month": "string", "date": "string", "utctime": "string"},
    )

    # Replace data gaps with NaN
    df_ace_mag.replace([-999.9, -100000], np.nan, inplace=True)
    df_ace_swp.replace([-9999.9, -100000], np.nan, inplace=True)

    # Set the indices of two dataframes to datetime objects/timestamps
    df_ace_mag.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_mag.year[i]}{df_ace_mag.month[i]}{df_ace_mag.date[i]}{df_ace_mag.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_mag.index))
        ]
    )

    df_ace_swp.index = np.array(
        [
            datetime.datetime.strptime(
                f"{df_ace_swp.year[i]}{df_ace_swp.month[i]}{df_ace_swp.date[i]}{df_ace_swp.utctime[i]}",
                "%Y%m%d%H%M",
            )
            for i in range(len(df_ace_swp.index))
        ]
    )

    # Combine the two dataframes in one single dataframe along the column/index
    df_ace = pd.concat([df_ace_mag, df_ace_swp], axis=1)

    # Remove the duplicate columns
    df_ace = df_ace.loc[:, ~df_ace.columns.duplicated()]

    # Compute the observation time in UNIX time
    # t_0_unix = datetime.datetime(1970, 1, 1)

    # time_o = (df_ace.index[-1].to_pydatetime() - t_0_unix).total_seconds()

    # Compute the dipole tilt angle of the earth
    # dipole_tilt = gp.recalc(time_o)

    # Save the flux data to the dataframe
    df_ace["flux"] = df_ace.np * df_ace.vp * 1e-3

    # Save the magnitude of magnetic field data to the dataframe
    df_ace["bm"] = np.sqrt(df_ace.bx_gsm**2 + df_ace.by_gsm**2 + df_ace.bz_gsm**2)

    # Compute the IMF clock angle and save it to dataframe
    df_ace["theta_c"] = np.arctan2(df_ace.by_gsm, df_ace.bz_gsm)

    # Compute the dynamic pressure of solar wind
    df_ace["p_dyn"] = 1.6726e-6 * 1.15 * df_ace.np * df_ace.vp**2

    # Get the unix time for all the time tags
    df_ace["unix_time"] = df_ace.index.astype(int) // 10**9

    # Compute the dipole tilt angle
    for i in range(len(df_ace)):
        # tilt_angle_gp = gp.recalc(df_ace.unix_time[i])
        tilt_angle_gp = gp.recalc(df_ace.unix_time.iloc[i])
        df_ace.loc[df_ace.index[i], "dipole_tilt"] = np.degrees(tilt_angle_gp)

    # Compute the dipole tilt angle correction, indegrees,  to be applied to the cusp locations
    # NOTE: The correctionvalue computed  here is based on the values given in Newell et al. (2006),
    # doi:10.1029/2006JA011731, 2006
    df_ace["dipole_tilt_corr"] = -0.046 * df_ace.dipole_tilt * 180 / np.pi

    # Compute the location of cusp based on different coupling equations
    df_ace["lambda_phi"] = (
        -3.65e-2
        * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4) ** (2 / 3)
        + 77.2
        + df_ace.dipole_tilt_corr
    )

    df_ace["lambda_wav"] = (
        -2.27e-3 * (df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 4)
        + 78.5
        + df_ace.dipole_tilt_corr
    )
    df_ace["lambda_vas"] = (
        -2.14e-4
        * df_ace.p_dyn ** (1 / 6)
        * df_ace.vp ** (4 / 3)
        * df_ace.bt
        * np.sin(df_ace.theta_c) ** 4
        + 78.3
        + df_ace.dipole_tilt_corr
    )
    df_ace["lambda_ekl"] = (
        -1.90e-3 * df_ace.vp * df_ace.bt * np.sin(df_ace.theta_c / 2.0) ** 2
        + 78.9
        + df_ace.dipole_tilt_corr
    )

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
        df_ace.index.values, df_ace.bm.values, "w-.", lw=lw, ms=ms, label=r"$|\vec{B}|$"
    )
    axs1.plot(df_ace.index.values, -df_ace.bm.values, "w-.", lw=lw, ms=ms)
    axs1.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.bm.isnull().all():
        axs1.set_ylim([-1, 1])
    else:
        axs1.set_ylim(-1.1 * np.nanmax(df_ace.bm), 1.1 * np.nanmax(df_ace.bm))

    axs1.set_xlim(df_ace.index.min(), df_ace.index.max())
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

    fig.suptitle("1 Day ACE Real Time Data", fontsize=22)

    # Density plot
    axs2 = fig.add_subplot(gs[1, 0], sharex=axs1)
    axs2.plot(
        df_ace.index.values,
        df_ace.np.values,
        color="bisque",
        ls="-",
        lw=lw,
        ms=ms,
        label=r"$n_p$",
    )
    axs2.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.np.isnull().all():
        axs2.set_ylim([-1, 1])
    else:
        axs2.set_ylim(0.9 * np.nanmin(df_ace.np), 1.1 * np.nanmax(df_ace.np))

    # lgnd2 = axs2.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd2.legendHandles[0]._sizes = [labelsize]
    axs2.set_ylabel(r"$n_p [1/\rm{cm^{3}}]$", fontsize=ylabelsize, color="bisque")

    # Speed plot
    axs3 = fig.add_subplot(gs[2, 0], sharex=axs1)
    axs3.plot(df_ace.index.values, df_ace.vp.values, "c-", lw=lw, ms=ms, label=r"$V_p$")
    axs3.axvspan(t1, t2, alpha=alpha, color=bar_color)

    if df_ace.vp.isnull().all():
        axs3.set_ylim([-1, 1])
    else:
        axs3.set_ylim(0.9 * np.nanmin(df_ace.vp), 1.1 * np.nanmax(df_ace.vp))

    # lgnd3 = axs3.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd3.legend_handles[0]._sizes = [labelsize]
    axs3.set_ylabel(r"$V_p [\rm{km/sec}]$", fontsize=ylabelsize, color="c")

    # Flux plot
    axs4 = fig.add_subplot(gs[3, 0], sharex=axs1)
    axs4.plot(
        df_ace.index.values, df_ace.flux.values, "w-", lw=lw, ms=ms, label=r"flux"
    )
    axs4.axhline(
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

    # lgnd4 = axs4.legend(fontsize=labelsize, loc="best", ncol=ncols)
    # lgnd4.legend_handles[0]._sizes = [labelsize]
    axs4.set_ylabel(
        r"~~~~Flux\\ $10^8 [\rm{1/(sec\, cm^2)}]$", fontsize=ylabelsize, color="w"
    )

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
    )
    axs5.plot(df_ace.index.values, df_ace.lambda_wav.values, "b-", lw=lw, ms=ms)
    axs5.plot(df_ace.index.values, df_ace.lambda_vas.values, "g-", lw=lw, ms=ms)
    axs5.plot(df_ace.index.values, df_ace.lambda_ekl.values, "m-", lw=lw, ms=ms)
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

    fig_name = "rt_sw_ace_parameters_2hr.png"

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
    return df_ace


# s.enter(0, 1, plot_figures_ace_1day, (s,))
# s.run()

# Print that the code has finished running and is waiting for the next update in 60 seconds
print(
    "Code execution for ace 1day data finished at (UTC):"
    + f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
)


# Display a progress bar for the next update

# s.enter(0, 1, plot_figures_ace, (s,))
# s.run()

if __name__ == "__main__":
    df_ace = plot_figures_ace()
