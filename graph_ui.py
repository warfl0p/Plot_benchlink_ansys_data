import os
import re
from pathlib import Path
from tkinter import *
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy.optimize import curve_fit
# from sklearn.linear_model import LinearRegression
# import seaborn as sns

main_window = Tk()
# photo = PhotoImage(file = "simulatie\icon.png")
# main_window.iconphoto(False, photo)
main_frame1 = Frame(main_window)
main_frame1.grid(row=0, column=0, padx=10, pady=1, sticky="NW")
main_frame2 = Frame(main_window)
main_frame2.grid(row=1, column=0, padx=10, pady=1, sticky="NW")
main_frame21 = LabelFrame(main_frame2, text="simulatie")
main_frame21.grid(row=0, column=0, padx=10, pady=5, sticky="NW")
main_frame22 = LabelFrame(main_frame2, text="data logger")
main_frame22.grid(row=1, column=0, padx=10, pady=5, sticky="NW")
main_frame3 = LabelFrame(main_window, text="extra")
main_frame3.grid(row=2, column=0, padx=10, pady=10, sticky="NW")

frame1 = LabelFrame(main_frame1, text='graph settings')
frame1.grid(row=0, column=0, padx=10, pady=10, sticky="NW")
frame3 = LabelFrame(main_frame1, text= 'file selection/creation')
frame3.grid(row=0, column=1, padx=10, pady=10, sticky="NW")

legend_sim_frame = Frame(main_frame21)
legend_sim_frame.grid(row=1, column=0, padx=10, pady=10)
legend_sim_entry = Frame(main_frame21)
legend_sim_entry.grid(row=1, column=1, padx=10, pady=10)
linetype_sim_frame = Frame(main_frame21)
linetype_sim_frame.grid(row=1, column=2, padx=10, pady=10)

legend_logger_frame = Frame(main_frame22)
legend_logger_frame.grid(row=1, column=3, padx=10, pady=10)
legend_logger_entry = Frame(main_frame22)
legend_logger_entry.grid(row=1, column=4, padx=10, pady=10)
linetype_logger_frame = Frame(main_frame22)
linetype_logger_frame.grid(row=1, column=6, padx=10, pady=10)
tc_frame = Frame(main_frame22)
tc_frame.grid(row=1, column=5, padx=10, pady=10)
title = StringVar()
aantalLijnen = StringVar()
legend = StringVar()
filetype = StringVar()
lines_to_skip_start = IntVar()
lines_to_skip_end= IntVar()
lines_to_skip_end.set(350)
filetype.set("pdf")
yaxis = StringVar()
yaxis.set("Temperature (K)")
linetype = StringVar()
dead_time = IntVar()  #50=115.2 seconds
dead_time.set(93)
linetypeLog = []
linetypeSim = []
thermocoupleList = []
filesLog = []
filesSim= []
def _quit():
    main_window.quit()
    main_window.destroy()

main_window.protocol("WM_DELETE_WINDOW", _quit)

def remove_text(event):
    # Remove the default text when the user clicks on the widget
    if title_box.get() == "Enter title here":
        title_box.delete(0, "end")

def placeWidgets(legendFrame, legendEntry, linetypeFrame, filesList, linetype):
    for file_path in filesList:
        result = file_path[file_path.rfind("/") + 1 :]
        entry_label = Label(legendFrame, text=result)
        entry_label.pack(pady=5)
        entry_entry = Entry(legendEntry)
        entry_entry.pack(pady=6)
        options = ["solid", "dotted", "dashed", "dashdot"]
        selected_option = StringVar()
        selected_option.set(options[0])
        entry_menu = OptionMenu(linetypeFrame, selected_option, *options)
        entry_menu.pack()
        linetype.append(selected_option)

def removeWidgets(legendFrame, legendEntry, linetypeFrame,files):
    files.clear()
    if legendFrame == legend_logger_frame:
        linetypeLog.clear()
        thermocoupleList.clear()
        for widget in tc_frame.winfo_children():
            widget.destroy()
    for widget in legendFrame.winfo_children():
        widget.destroy()
    for widget in legendEntry.winfo_children():
        widget.destroy()
    for widget in linetypeFrame.winfo_children():
        widget.destroy()

def browseSimFiles():
    global filesSim, linetypeSim , linetypeSim, newFilesSim
    # Allow user to select one or more files
    currentDirectory = os.getcwd()
    newFilesSim = fd.askopenfilenames(
        initialdir=os.path.join(currentDirectory, "simulatie/Resultaten HPC"))
    filesSim.extend(newFilesSim)
    clear_sim_button = Button(frame3, text="clear simulation files", command=lambda: removeWidgets(legend_sim_frame, legend_sim_entry, linetype_sim_frame, filesSim))
    clear_sim_button.grid(column=1, row=0, sticky="NW")
    placeWidgets(legend_sim_frame, legend_sim_entry, linetype_sim_frame, newFilesSim, linetypeSim)

def browseLogFiles(duplicates):
    global filesLog, linetypeLog, dfs, thermocoupleList, newFilesLog
    # Allow user to select one or more files
    if duplicates == True:
        try:
            filesLog.extend(newFilesLog)
        except:
            print("there is no previous selection")
    else:
        currentDirectory = os.getcwd()
        newFilesLog = fd.askopenfilenames(initialdir = "benchlink/datalogger_files")
        filesLog.extend(newFilesLog)
    dfs = readDataLog(filesLog)
    placeWidgets(
        legend_logger_frame,
        legend_logger_entry,
        linetype_logger_frame,
        newFilesLog,
        linetypeLog)
    clear_logger_button = Button(
        frame3,
        text="clear datalogger files",
        command=lambda: removeWidgets(legend_logger_frame, legend_logger_entry, linetype_logger_frame, filesLog))
    clear_logger_button.grid(column=1, row=1, sticky="NW")
    for i in range(len(newFilesLog)):
        selected_tc = StringVar()
        tc_menu = OptionMenu(tc_frame, selected_tc, *colNames)
        tc_menu.pack()
        selected_tc.set(colNames[-1])
        thermocoupleList.append(selected_tc)

def retrieve_values(dataType):
    # Retrieve the values from the Entry widgets
    if dataType == "simulatie":
        frame = legend_sim_entry
    else:
        frame = legend_logger_entry
    legendvalslist = []
    for widget in frame.winfo_children():
        if isinstance(widget, Entry):
            legendvalslist.append(widget.get())
    return legendvalslist

def readLogData(csvfile, encodetype):
    df = pd.read_csv(csvfile, usecols=[0], encoding=encodetype, delimiter=';')
    rowsToSkip = df.loc[df[df.columns[0]] == "Scan"].index[0] + 1
    df = pd.read_csv(csvfile, skiprows=rowsToSkip, encoding=encodetype, delimiter=';')
    return df

def dropAlarmCols(df):
    global colNames
    colNames = list(df.columns)
    indexRemoveCol = []
    for i in range(3, len(colNames)):
        if i % 2 == True:
            indexRemoveCol.append(colNames[i])
    df = df.drop(columns=indexRemoveCol)
    try:
        df = df.drop(
            columns=["102 (C)", "104 (C)", "117 (C)", "120 (C)","121 (ADC)","122 (ADC)"]
        )  # deze thermocouples werken niet en worden dus uit dataframe verwijderd
    except:
        print(
            "ERROR: kan gekozen columns niet droppen omdat ze niet bestaan in dataframe"
        )
    return df

def formatTimecol(df):
    df.insert(2, "Time_seconds", None)  # voegt colomn looptijd in seconden toe
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%Y %H:%M:%S:%f")
    df["Time_seconds"] = (df["Time"] - df["Time"].iat[0]).apply(
        lambda x: x.total_seconds()
    )
    df = df.drop(columns="Time")
    return df

def readDataSim(csvfile):
    listcolnames = []
    df = pd.read_csv(csvfile, nrows=3)
    input_string = df.iloc[1, 0]
    for text in re.findall('"([^"]*)"', input_string):  # pakt string tussen " " voor column naam
        listcolnames.append(text)
    df = pd.read_csv(csvfile, names=listcolnames, delim_whitespace=True,
        skiprows=lambda x: x % 300 != 0 or x < 3,)  # skipt de eerste drie rijen, en leest 1 op 300 rijen
    return df

def avgColumn(df):
    global colNames
    # cols = df[['101 (C)','103 (C)','105 (C)','106 (C)','107 (C)','108 (C)','109 (C)','110 (C)','111 (C)','114 (C)','115 (C)','118 (C)']]
    cols =df[['101 <00> (C)','103 <02> (C)','105 <04> (C)','106 <05> (C)','107 <06> (C)','108 <07> (C)','109 <08> (C)','110 <09> (C)','111 <10> (C)','112 <11Outside> (C)','113 <12> (C)','114 <13> (C)','115 <14> (C)','116 <15> (C)','118 <17> (C)','119 <18Outside> (C)']]
    # cols = df.iloc[:, 1:-1]
    # row_avg = cols.mean(axis=1)
    df['average'] = cols.mean(axis=1)
    colNames = list(df.columns)
    return df

def readDataLog(csvfiles):
    global dfs
    dfs = []
    for csvfile in csvfiles:
        try:
            df = readLogData(csvfile, "utf8")  # dataframe uitlezen
        except:
            df = readLogData(csvfile, "UTF-16 LE")
        df = dropAlarmCols(df)  # enkele bewerkingen op dataframe
        df = formatTimecol(df)
        df = avgColumn(df)
        dfs.append(df)
    return dfs

def extraCalcs():
    dfcols = pd.DataFrame()
    for i in range(len(dfs)):
        dfcols[f'{thermocoupleList[i].get()}{i}']=dfs[i][thermocoupleList[i].get()][:6900]
    dfcols['std'] = dfcols.apply(lambda x: x.std(), axis=1)
    print('stdev: ','\n',dfcols)
    print('average std:',dfcols['std'].mean())
labelcount=0

def plot_averageDF():
    global df_avg, labelcount
    try:
        df_avg = pd.DataFrame(columns=dfs[0].columns)
        max_len = max(len(df) for df in dfs)
        min_len = min(len(df) for df in dfs)
        for col in df_avg.columns:
            col_avg = []
            for i in range(min_len):
                col_sum = sum(df[col][i] for df in dfs if len(df) > i)
                col_avg.append(col_sum / len([df for df in dfs if len(df) > i]))
            if max_len > min_len:
                for i in range(min_len, max_len):
                    col_sum = sum(df[col][i] for df in dfs if len(df) > i)
                    col_avg.append(col_sum / len([df for df in dfs if len(df) > i]))
            df_avg[col] = col_avg
        # df_average.to_excel('no fins average.xlsx')                #zet eerst geselecteerde df naar excel 
        plt.plot(df_avg['Time_seconds'], df_avg['average'], label="Average")
        # plt.title("Temperature Decrease Rate")
    except:
        combined_data = pd.DataFrame()
        for i in range(len(filesSim)):
            df = readDataSim(filesSim[i])

            time_column = df.columns[0]
            data_column = df.columns[1]

            # Rename the columns for consistency
            df = df.rename(columns={time_column: 'time', data_column: 'data'})

            # Append the data to the combined_data DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)
        # Group the data by the time column and calculate the average of the data column
        average_data = combined_data.groupby('time')['data'].mean().reset_index()
        list=['fins','no fins']
        
        # Plot the average data
        plt.plot(average_data['time'], average_data['data'],label=list[labelcount])
        labelcount=1
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.legend()
    plt.show()

def calc_derivative():
    dt = 10
    for i in range(len(dfs)):
        dfs[i]['rolling_average'] = dfs[i]['average'].rolling(window=20).mean()
        dfs[i]['Time_derivative'] = np.gradient(dfs[i]['rolling_average'], dt)
    dtemp = dfs[0]['Time_derivative'][400:]
    x=  dfs[0]["Time_seconds"][400:]
    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(b * t), x, dtemp,p0=[-0.01,-0.00005])
    x_new = np.linspace(x.min(), x.max(), len(dtemp))
    a, b = popt[0], popt[1]             #obtain coefficients for the trend line
    y_new = a * np.exp(b * x_new)
    residuals = dtemp- y_new
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((dtemp-np.mean(dtemp))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)
    plt.plot(x, dtemp, label="Temperature derivative")
    plt.plot(x_new, y_new, label="Trendline",color="black")
    # plt.title("Temperature Decrease Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature Change(K/s)")
    plt.legend()
    plt.savefig(f"simulatie/plots-sim-{filetype.get()}/Temperature decrease.{filetype.get()}",bbox_inches='tight')
    dfs[0].to_csv('figuren/no fins df.csv',index=False)
    plt.show()
    return dfs

def calc_trendline_sim():
    legendvalslistsim = retrieve_values("simulatie")
    df = readDataSim(filesSim[0])
    df.iloc[:, [1]] = df.iloc[:, [1]].shift(dead_time.get())
    selected_rows_trend  = df[(df['flow-time'] <= 190*10) & (df['flow-time'] >= 50*10)]
    selected_rows  = df[(df['flow-time'] <= lines_to_skip_end.get()*10) & (df['flow-time'] >= lines_to_skip_start.get()*10)]
    x=selected_rows["flow-time"]
    y=selected_rows.iloc[:, [1]]
    x_trend = selected_rows_trend["flow-time"]
    y_trend = selected_rows_trend.iloc[:, [1]]
    x_trend = x_trend.values.reshape(-1, 1)  # reshape(-1, 1) converts a 1D array to a 2D array
    y_trend = y_trend.values
    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(x_trend,   y_trend)

    # Get the coefficients of the linear regression model
    a = model.coef_[0]
    b = model.intercept_
    print(f'y = {a}x + {b}')
    # Plot the data and the trendline
    plt.plot(x, y, label=legendvalslistsim[0], color="#1f77b4")
    plt.plot(x_trend, model.predict(x_trend), label=f'y = {a[0]:.2f}x + {b[0]:.2f}', linestyle='dashed', color="#d62728")
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.savefig(f"simulatie/plots-sim-png/Enter title here.png",bbox_inches='tight')
    plt.savefig(f"simulatie/plots-sim-{filetype.get()}/{title.get()}.{filetype.get()}",bbox_inches='tight')
    plt.show()

def calc_trendline_log():
    legendvalslistlog = retrieve_values("datalogger")
    y = dfs[0][thermocoupleList[0].get()][50:190]+273.15
    x=  dfs[0]["Time_seconds"][50:190]
    x = x.values.reshape(-1, 1)  # reshape(-1, 1) converts a 1D array to a 2D array
    y = y.values
    # Create a linear regression model and fit it to the data
    model = LinearRegression()
    model.fit(x, y)

    # Get the coefficients of the linear regression model
    a = model.coef_[0]
    b = model.intercept_
    print(f'y = {a}x + {b}')
    # Plot the data and the trendline
    plt.plot(dfs[0]["Time_seconds"][lines_to_skip_start.get():lines_to_skip_end.get()], dfs[0][thermocoupleList[0].get()][lines_to_skip_start.get():lines_to_skip_end.get()]+273.15,
             label=legendvalslistlog[0],color="#ff7f0e")
    plt.plot(x, model.predict(x), label=f'y = {a:.2f}x + {b:.2f}', linestyle='dashed',color="#2ca02c")
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.savefig(f"simulatie/plots-sim-png/Enter title here.png",bbox_inches='tight')
    if title.get()=='':
        plt.savefig(f"simulatie/plots-sim-{filetype.get()}/no title.{filetype.get()}",bbox_inches='tight')
    plt.savefig(f"simulatie/plots-sim-{filetype.get()}/{title.get()}.{filetype.get()}",bbox_inches='tight')
    plt.show()

def plotMeasurement():
    legendvalslistsim = retrieve_values("simulatie")
    legendvalslistlog = retrieve_values("datalogger")
    fig, ax = plt.subplots(1)
    # ax.grid(True)
    # fig.set_size_inches(18.5, 10.5, forward=True)
    # try:
    for i in range(len(filesSim)):
        df = readDataSim(filesSim[i])
        df.iloc[:, [1]] = df.iloc[:, [1]].shift(dead_time.get())        
        selected_rows  = df[(df['flow-time'] <= lines_to_skip_end.get()*10) & (df['flow-time'] >= lines_to_skip_start.get()*10)]
        plt.plot(
            selected_rows["flow-time"],
            selected_rows.iloc[:, [1]],
            label=legendvalslistsim[i],
            linestyle=linetypeSim[i].get())
    # except:
        # print("geen/fout met simulatie files")
    # try:
    dfs = readDataLog(filesLog)
    for i in range(len(dfs)):
        # dfs[i][thermocoupleList[i].get()].add(273.15)
        plt.plot(
        dfs[i]['Time_seconds'][lines_to_skip_start.get():lines_to_skip_end.get()],
        dfs[i][thermocoupleList[i].get()][lines_to_skip_start.get():lines_to_skip_end.get()] + 273.15,
        label=legendvalslistlog[i],
        linestyle=linetypeLog[i].get())
    # except:
    #     print("geen/fout met datalogger files")
    plt.xlabel("Time (s)")  # as titels toevoegen
    plt.ylabel(yaxis.get())
    plt.legend()
    # ax.set_ylim(ymin=15)                              # startwaarde op y-as instellen (optioneel)
    plt.title(title.get())
    # plt.savefig(f"plots-sim-png/Enter title here.png",bbox_inches='tight')
    # plt.savefig(f"plots-sim-{filetype.get()}/{title.get()}.{filetype.get()}",bbox_inches='tight')
    plt.savefig(f"simulatie/plots-sim-png/Enter title here.png",bbox_inches='tight')
    if title.get()=='':
        plt.savefig(f"simulatie/plots-sim-{filetype.get()}/no title.{filetype.get()}",bbox_inches='tight')
    plt.savefig(f"simulatie/plots-sim-{filetype.get()}/{title.get()}.{filetype.get()}",bbox_inches='tight')
    # plt.show()
def saveGraph():
    plt.savefig(f"simulatie/plots-sim-{filetype.get()}/savegraph.pdf",bbox_inches='tight')
    

plot_button = Button(main_frame3, text="Plot graph", command=plotMeasurement)
plot_button.grid(column=0, row=0, sticky="NW")
save_button = Button(main_frame3, text="Save graph", command=saveGraph)
save_button.grid(column=0, row=1, sticky="NW")
plot_extra_button = Button(main_frame3, text="Plot extra", command=extraCalcs)
plot_extra_button.grid(column=1, row=0, sticky="NW")
average_button = Button(main_frame3, text="save average", command=plot_averageDF)
average_button.grid(column=1, row=1, sticky="NW")
trendline_button = Button(main_frame3, text="plot trend sim", command=calc_trendline_sim)
trendline_button.grid(column=2, row=1, sticky="NW")
trendline_button = Button(main_frame3, text="plot trend log", command=calc_trendline_log)
trendline_button.grid(column=3, row=1, sticky="NW")
plot_derivative_button = Button(main_frame3, text="Plot derivative", command=calc_derivative)
plot_derivative_button.grid(column=2, row=0, sticky="NW")
browse_sim_button = Button(frame3, text="select simulation files", command=browseSimFiles)
browse_sim_button.grid(column=0, row=0, sticky="NW")
select_logger_button = Button(frame3, text="select datalogger files", command=lambda: browseLogFiles(False))
select_logger_button.grid(column=0, row=1, sticky="NW")
duplicate_logger_button = Button(frame3, text="duplicate last selection", command=lambda: browseLogFiles(True))
duplicate_logger_button.grid(column=0, row=2, sticky="NW")

title_label = Label(frame1, text="title: ")
title_label.grid(row=0, column=0, padx=10, pady=10)

line_start_label = Label(frame1, text="from line: ")
line_start_label.grid(row=2, column=0, sticky="NE")
line_end_label = Label(frame1, text="to: ")
line_end_label.grid(row=3, column=0, sticky="NE")
dead_label = Label(frame1, text="dead time: ")
dead_label.grid(row=4, column=0)

title_box = Entry(frame1, textvariable=title, width=25)
title_box.insert(0, "Enter title here")
title_box.grid(column=1, row=0)
title_box.bind("<Button-1>", remove_text)

filetype_menu = OptionMenu(frame1, filetype, "pdf", "png")
filetype_menu.grid(column=0, row=1, sticky="NW")

skiplines_start_entry = Entry(frame1, textvariable=lines_to_skip_start, width=25)
skiplines_start_entry.grid(column=1, row=2, sticky="NW")
skiplines_end_entry= Entry(frame1, textvariable=lines_to_skip_end, width=25)
skiplines_end_entry.grid(column=1, row=3, sticky="NW")
deadtimeEntry= Entry(frame1, textvariable=dead_time, width=25)
deadtimeEntry.grid(column=1, row=4, sticky="NW")
yaxis_menu = OptionMenu(frame1, yaxis, "Temperature (K)", r"Heat Flux (W/m$^2$)")
yaxis_menu.grid(column=1, row=1, sticky="NW")
# cap = tkcap.CAP(main_window)     # master is an instance of tkinter.Tk
# cap.capture('screenshot')
main_window.mainloop()
