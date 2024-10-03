import importlib
import time
import datetime
import sw_rt_ts_dwld_upld_dscovr_07days_automated as sw_rt_ts_dwld
import sw_rt_ts_mp4_dscovr as sw_rt_ts_mp4

importlib.reload(sw_rt_ts_dwld)
importlib.reload(sw_rt_ts_mp4)


# Run the rest of the code every 60 minutes
while True:
    time_code_start = time.time()
    number_of_days = 28
    df_dscovr_hc = sw_rt_ts_dwld.plot_figures_dsco_07days(number_of_days=number_of_days)
    sw_rt_ts_mp4.make_gifs(
        number_of_files=number_of_days * 24,
        image_path="figures/historical/dscovr/07days/",
        vid_type="mp4",
    )

    # plt.close("all")
    # Copy the gif file to google drive
    # os.system("cp /home/cephadrius/Desktop/git/qudsiramiz.github.io/images/moving_pictures/*" +
    #           "/home/cephadrius/google-drive/Studies/Research/bu_research/dxl/figures/vids/")
    print(f"Time taken: {round(time.time() - time_code_start, 2)} seconds")
    print(
        f"Waiting for the next update at: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )
    # Every 5 minutes print the time to say how much time is left
    for i in range(0, 60, 5):
        print(f"{60-i} minutes left until the next Update")
        time.sleep(300)
