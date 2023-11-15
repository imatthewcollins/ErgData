import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

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
        seconds = float(time_str_parts[1])
        return pd.Timedelta(minutes=minutes, seconds=seconds)
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
        # print(timedelta_obj.total_seconds(), minutes, seconds)
        seconds = round(seconds, 1)  # Round to one decimal place
        return f'{int(minutes):02}:{seconds:04.1f}'
    else:
        return ''

# --------------------------------------

def seconds_to_mmss(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f'{int(minutes):02}:{seconds:04.1f}'

# --------------------------------------

def plot_split_versus_data_for_athlete(name, df):
    current_directory = os.getcwd()

    x = df['Dates']
    y = df['AverageTimeInSeconds']

    formatted_name = name.lower().replace(" ", "_")
    subfolder_path = os.path.join(current_directory, 'graphs')
    athlete_path = os.path.join(subfolder_path, 'athlete')
    file_name = f'{formatted_name}.png'

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(athlete_path):
        os.makedirs(athlete_path)
    file_path = os.path.join(athlete_path, file_name)

    plt.figure(figsize=(20,10))
    plt.plot(x,y,label=name, marker='o', color='b')

    # Set equidistant y-tick positions (e.g., every 60 seconds)
    min_time_seconds = min(y)
    max_time_seconds = max(y)
    y_ticks_seconds = range(int(min_time_seconds), int(max_time_seconds)+1, 1)

    # Convert y-tick positions to mm:ss.s format
    y_labels = [seconds_to_mmss(y_tick) for y_tick in y_ticks_seconds]

    plt.xticks(rotation=45, ha='right')
    plt.yticks(y_ticks_seconds, y_labels)

    plt.xlabel('Date')
    plt.ylabel('Split')
    plt.title(f'{name} Erg Results')

    for i, j, label in zip(x,y, df['AverageTimeString']):
        date = i.strftime('%Y-%m-%d')
        plt.annotate(f'{label}\n{date}', (i,j), textcoords='offset points', xytext=(10, 5), ha='left', rotation=0)

    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

# --------------------------------------

def plot_week_data(df, week_number, workout, workout_date):
    # # Plotting
    x = df['Name']
    y = df['AverageTimeInSeconds']

    formatted_workout_date = workout_date.replace("-", "")[2:]
    formatted_workout = workout.split(" ")[0].replace(".", "_")
    formatted_file = f"{formatted_workout_date}_{formatted_workout}"

    current_directory = os.getcwd()
    subfolder_path = os.path.join(current_directory, 'graphs')
    team_path = os.path.join(subfolder_path, 'team')
    file_name = f'{formatted_file}.png'

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    if not os.path.exists(team_path):
        os.makedirs(team_path)
    file_path = os.path.join(team_path, file_name)


    plt.figure(figsize=(20,10))
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
    plt.savefig(file_path)
    plt.close()

# --------------------------------------

def plot_athletes_vs_split_for_workouts(weeks):
    for week_number, week in enumerate(weeks):
        week_df = df[(df['Dates'] >= week[0]) & (df['Dates'] <= week[1])]
        workout = week[2]
        actual_workout_date = week[3]
        plot_week_data(week_df, week_number + 1, workout, actual_workout_date)

# --------------------------------------

def merge_data(old_essential_df, new_data_df):
    common_ids = set(old_essential_df['ID']) & set(new_data_df['ID'])
    new_data_df_filtered = new_data_df[~new_data_df['ID'].isin(common_ids)]

    new_master_df = pd.concat([old_essential_df, new_data_df_filtered], ignore_index=True)
    new_master_df.to_excel('data/updated_master.xlsx', index=False)

    return new_master_df


# --------------------------------------
# PREPARING THE DATA
# --------------------------------------

# Import data

old_data_file = "data/essential_old_data.xlsx"
new_data_file = "data/231115_new_data.xlsx"

old_data_df = pd.read_excel(old_data_file)
new_data_df = pd.read_excel(new_data_file)
df = merge_data(old_data_df, new_data_df)

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Drop redundant columns
columns_to_remove = ['ID', 'Start time', 'Completion time', 'Email', 'Last modified time']
df = df.drop(columns=columns_to_remove)

# Clean the data in the time columns
time_columns = ['ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the 4k piece (mm:ss.s)', 'ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the 2k piece (mm:ss.s)', 'ENTER DATA AS MM:SS.S. Enter your AVERAGE SPLIT for the 1k piece (mm:ss.s)']
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
week6 = ('2023-10-31', '2023-11-02', '4k_2k_1k', '2023-11-01')
week7 = ('2023-11-14', '2023-11-16', '3x10min 4,3,2,1 r20,22,24,26', '2023-11-15')

weeks = [week1, week2, week3, week4, week5, week6, week7]
# weeks = [week7]

plot_athletes_vs_split_for_workouts(weeks)


# df = df[df['Dates'] < '2023-10-31']
# unique_names = df['Name'].unique()
# columns_of_interest = ['Name', 'Dates', 'AverageTimeString', 'AverageTimeInSeconds']

# for name in unique_names:
#     filtered_df = df[df['Name'] == name]
#     filtered_df = filtered_df.loc[:, columns_of_interest]
#     filtered_df.sort_values(by='Dates', ascending=True, inplace=True)
#     plot_split_versus_data_for_athlete(name, filtered_df)

