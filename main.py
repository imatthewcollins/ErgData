import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import math


master_file = "data/master.xlsx"
new_data_file = "data/new_data.xlsx"
run_master = True

# --------------------------------------

def validate_and_convert(time):
    # Define a regular expression pattern for mm:ss.s format
    time_str = str(time)
    pattern = r'^\d{2}:\d{2}.\d$'
    
    if re.match(pattern, time_str):
        return time_str
    else:
        if time_str == 'nan':
            return np.nan
        else:
            new_time_str = ''
            if ':' in time_str:
                split_time = time_str.split(':')

                # check length of split_time
                if len(split_time) == 2:
                    if len(split_time[0]) == 1:
                        new_time_str += '0' + split_time[0] + ':'

                        if '.' in split_time[1]:
                            split_ms = split_time[1].split('.')
                            if len(split_ms) == 2:
                                if len(split_ms[0]) == 1:
                                    new_time_str += '0' + split_ms[0]
                                else:
                                    seconds = int(split_ms[0])
                                    if seconds < 0 or seconds > 59:
                                        return np.nan
                                    else:
                                        new_time_str += split_ms[0]

                                if len(split_ms[1]) >= 1:
                                    milliseconds = split_ms[1][0]
                                    new_time_str += '.' + milliseconds

                        else:
                            new_time_str += split_time[1] + '.0'
                    
                    elif len(split_time[0]) == 2:
                        if '.' not in split_time[1]:
                            new_time_str += split_time[0] + ':' + split_time[1] + '.0' 
                        else:
                            split_ms = split_time[1].split('.')
                            if len(split_ms[0]) < 2:
                                new_time_str += split_time[0] + ':0' + split_ms[0] + '.' + split_ms[1]
                    
                elif len(split_time) > 2:
                    if split_time[0] == '00':
                        new_time_str = split_time[1] + ':' + split_time[2][:4]
                    else:
                        new_time_str = split_time[0] + ':' + split_time[1] + '.' + split_time[2]

            elif '.' in time_str:
                split_time = time_str.split('.')
                if len(split_time) == 3:
                    if len(split_time[0]) == 1:
                        new_time_str += '0' + split_time[0]
                    else:
                        new_time_str += split_time[0]
                    new_time_str += ':' + split_time[1] + '.' + split_time[2][0]

            else:
                print('---------------')
                print('no condition met!', time_str)
                print('---------------')

            if re.match(pattern, new_time_str):
                return new_time_str
            else:
                print('---------------')
                print(new_time_str)
                print('---------------')
                return np.nan


# --------------------------------------

def time_str_to_timedelta(time_input):
    time = str(time_input)
    try:
        time_str_parts = time.split(':')
        minutes = int(time_str_parts[0])
        seconds, milliseconds = map(int, time_str_parts[1].split('.'))
        return pd.Timedelta(minutes=minutes, seconds=seconds, milliseconds=milliseconds)
    except ValueError:
        return pd.NaT

# --------------------------------------

def time_str_to_numerical(time):
    time_str = str(time)
    try:
        time_str_parts = time_str.split(':')
        minutes_in_seconds = int(time_str_parts[0]) * 60
        seconds = float(time_str_parts[1])
        total_seconds = minutes_in_seconds + seconds
        return total_seconds
    except ValueError:
        return pd.NaT

# --------------------------------------

def format_timedelta(timedelta_obj):
    if pd.notna(timedelta_obj):
        minutes, seconds = divmod(timedelta_obj.total_seconds(), 60)
        seconds = round(seconds, 1)  # Round to one decimal place
        return f'{int(minutes):02}:{seconds:04.1f}'
    else:
        return ''

# --------------------------------------

def seconds_to_mmss(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f'{int(minutes):02}:{seconds:04.1f}'

# --------------------------------------

def plot_week_data(df, week_number, workout, workout_date):
    # # Plotting
    x = df['Name']
    y = df['AverageTimeInSeconds']

    plt.figure(figsize=(10,10))
    plt.plot(x,y,label=f'Week {week_number} Results', marker='o', color='b')

    # Set equidistant y-tick positions (e.g., every 60 seconds)
    min_time_seconds = min(y)
    max_time_seconds = max(y)
    y_ticks_seconds = range(int(min_time_seconds), int(max_time_seconds)+1, 1)
    # y_ticks_seconds = range(int(min_time_seconds // 60) * 60, int(max_time_seconds // 60) * 60 + 60, 60)

    # Convert y-tick positions to mm:ss.s format
    y_labels = [seconds_to_mmss(y_tick) for y_tick in y_ticks_seconds]

    # y_labels = week1_df['AverageTimeString']
    plt.xticks(rotation=45, ha='right')
    plt.yticks(y_ticks_seconds, y_labels)

    plt.xlabel('Name')
    plt.ylabel('Split')
    plt.title(f'{workout_date} | Week {week_number} Erg Results: {workout}')

    for i, j, label in zip(x,y, df['AverageTimeString']):
        plt.annotate(label, (i,j), textcoords='offset points', xytext=(-5, 5), ha='right', rotation=-15)

    plt.legend()
    plt.grid(True)
    plt.show()

# --------------------------------------

def plot_athletes_vs_split_for_workouts(weeks):
    for week_number, week in enumerate(weeks):
        week_df = df[(df['Dates'] >= week[0]) & (df['Dates'] <= week[1])]
        workout = week[2]
        actual_workout_date = week[3]
        plot_week_data(week_df, week_number + 1, workout, actual_workout_date)

# --------------------------------------
# PREPARING THE DATA
# --------------------------------------

# Import data
if run_master:
    df = pd.read_excel(master_file)
else:
    df = pd.read_excel(new_data_file)

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Drop redundant columns
columns_to_remove = ['ID', 'Start time', 'Completion time', 'Email', 'Last modified time']
df = df.drop(columns=columns_to_remove)

# Clean the data in the time columns
time_columns = ['ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the r20 piece (mm:ss.s)', 'ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the r22 piece (mm:ss.s)', 'ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the r24 piece (mm:ss.s)']
for col in time_columns:
    df[col] = df[col].apply(validate_and_convert)
    df[col] = df[col].apply(time_str_to_timedelta)

df['AverageTimeTimeDelta'] = df[time_columns].mean(axis=1, skipna=True)

df['AverageTimeString'] = df['AverageTimeTimeDelta'].apply(format_timedelta)

df['AverageTimeInSeconds'] = df['AverageTimeString'].apply(time_str_to_numerical)

# Sort the DataFrame by the 'NumericalTime' column in ascending order (fastest to slowest)
df.sort_values(by='AverageTimeInSeconds', ascending=True, inplace=True)

# Filter the DataFrame to remove rows with missing values
df = df.dropna(subset=['AverageTimeInSeconds'])

# Convert the 'Dates' column to datetime format
df['Dates'] = pd.to_datetime(df['Enter today\'s date'])


# --------------------------------------
# PLOTTING DATA
# --------------------------------------

# # Filter by a date range
week1 = ('2023-09-12', '2023-09-19', '3x8min r20,22,24', '2023-09-13')
week2 = ('2023-09-19', '2023-09-21', '2x12min r20,22,24', '2023-09-20')
week3 = ('2023-09-26', '2023-09-28', '3x9min r20,22,24', '2023-09-27')
week4 = ('2023-10-03', '2023-10-05', '2x13.5min r20,22,24', '2023-10-04')
week5 = ('2023-10-10', '2023-10-12', '3x10min r20,22,24', '2023-10-11')

weeks = [week1, week2, week3, week4, week5]

# plot_athletes_vs_split_for_workouts(weeks)


unique_names = df['Name'].unique()
columns_of_interest = ['Name', 'Dates', 'AverageTimeString']

for name in unique_names:
    filtered_df = df[df['Name'] == name]
    filtered_df = filtered_df.loc[:, columns_of_interest]
    filtered_df.sort_values(by='Dates', ascending=True, inplace=True)
    print(filtered_df)