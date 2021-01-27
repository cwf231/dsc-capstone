################################################################################
# Scraping data from files

# To run this file: 
# It is not necessary to download any files manually.
# If the directory './data' does not exist, this file will download and unzip
# the necessary files and save them to the directory.
# Once scraped and transformed to csv files, the original data will be removed.

# If you prefer to download the files manually:

# 1. Downloaded from https://www.retrosheet.org/game.htm

# > Regular season event files by decade:
# - 1910-1919
# - 1920-1929
# - 1930-1939
# - 1940-1949
# - 1950-1959
# - 1960-1969
# - 1970-1979
# - 1980-1989
# - 1990-1999
# - 2000-2009
# - 2010-2019

# 2. Unzipped into the data folder `('./data')` so the tree structure is:

# └───dsc-capstone
#     ├───data
#     │   ├───1910seve
#     │   ├───1920seve
#     │   ├───1930seve
#     │   ├───1940seve
#     │   ├───1950seve
#     │   ├───1960seve
#     │   ├───1970seve
#     │   ├───1980seve
#     │   ├───1990seve
#     │   ├───2000seve
#     │   └───2010seve


# There should be numerous files within each *decade folder.*

# > File types:
# - `{year}{team}.EVL` (event files)
# - `{year}.EBL` (box score event files)
# - `{year}.EDL` (deduced event files)
# - `{team}{year}.ROS` (roster files)
#   - we will ignore the roster files
# - `{team}{year}` (teams - code / league / city / name)

# ########## Tables ##########

# ########## GAMES table ##########
# [game_id, visiting_team_id, home_team_id, 
# site_id, date, dblhdr_number, day_night, 
# temp, wind, field_cond, precip, time_of_game, attendance]

# ########## EVENTS table ##########
# [game_id, inning_num, inning_half, hitter_id, pitcher_id, outcome]

# ########## TEAMS table ##########
# [team_id, year, league, location, name]

################################################################################
import pandas as pd
import os
import shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def create_data_directory(start_year):
	"""
	Downloads, unzips, and saves files into ./data directory.
	start_year (integer) is the year to begin the data from.
		full dataset: 
			1910
		available choices: 
			[1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
	"""
    
	file_lst = [f'{str(y)}seve' for y in range(start_year, 2020, 10)]
	url_lst = [os.path.join('https://www.retrosheet.org/events/', f'{f}.zip')
			   for f in file_lst]

	for f, zip_url in zip(file_lst, url_lst):
		print(f'Downloading/extracting {f} from:\t{zip_url}')
		with urlopen(zip_url) as zipresp:
			with ZipFile(BytesIO(zipresp.read())) as zfile:
				zfile.extractall(f'./data/{f}')


def process_event_file(f_path):
    """Returns two lists: `games_table_rows`, `events_table_rows`"""
    
    def set_new_game():
        return {
            'game_id': '', 
            'visiting_team_id': '', 
            'home_team_id': '', 
            'site_id': '', 
            'date': '', 
            'dblhdr_number': '', 
            'day_night': '', 
            'temp': '', 
            'wind': '', 
            'field_cond': '', 
            'precip': '', 
            'time_of_game': '', 
            'attendance': ''
        }

    games_table_rows = []
    events_table_rows = []
    with open(f_path, 'r') as fh:
        current_game = set_new_game()
        current_pitchers = {
            'visitor': '',
            'home': ''
        }
        for line in fh:
            elements = line.strip().split(',')
            marker = elements[0]

            # Check if we hit a new game.
            if marker == 'id': # start a new game
                if current_game['game_id'] != '':
                    games_table_rows.append(current_game) # dump previous game data
                    current_game = set_new_game()
                current_game_id = elements[-1] # set new game_id
                current_game['game_id'] = current_game_id

            # Check for game elements.
            if marker == 'info':
                data = elements[1]

                if data == 'visteam':
                    current_game['visiting_team_id'] = elements[-1]
                elif data == 'hometeam':
                    current_game['home_team_id'] = elements[-1]
                elif data == 'site':
                    current_game['site_id'] = elements[-1]
                elif data == 'date':
                    current_game['date'] = elements[-1]
                elif data == 'number':
                    current_game['dblhdr_number'] = elements[-1]
                elif data == 'daynight':
                    current_game['day_night'] = elements[-1]
                elif data == 'temp':
                    current_game['temp'] = elements[-1]
                elif data == 'windspeed':
                    current_game['wind'] = elements[-1]
                elif data == 'fieldcond':
                    current_game['field_cond'] = elements[-1]
                elif data == 'precip':
                    current_game['precip'] = elements[-1]
                elif data == 'timeofgame':
                    current_game['time_of_game'] = elements[-1]
                elif data == 'attendance':
                    current_game['attendance'] = elements[-1]

            # Get starting pitchers or pitching substitutions.
            if (marker == 'start' or marker == 'sub') and elements[-1] == '1':
                if elements[-3] == '0':
                    current_pitchers['visitor'] = elements[1]
                else:
                    current_pitchers['home'] = elements[1]

            # Get the events from a single line.
            if marker == 'play' and elements[-1] != 'NP': # filer "no play" entries
                home_at_bat = elements[2] == '1'
                current_event = {
                    'game_id': current_game_id, 
                    'inning_num': elements[1], 
                    'inning_half': elements[2], 
                    'hitter_id': elements[3], 
                    'pitcher_id': current_pitchers['visitor'] if home_at_bat \
                                  else current_pitchers['home'], 
                    'outcome': elements[-1]
                }
                events_table_rows.append(current_event)
                
    return games_table_rows, events_table_rows


def process_team_file(f_path):
    """Returns list: `team_table_rows`"""
    team_table_rows = []
    with open(f_path, 'r') as fh:
        for line in fh:
            elements = line.strip().split(',')
            r = {
                'team_id': elements[0], 
                'year': f_path[-4:], 
                'league': elements[1], 
                'location': elements[-2], 
                'name': elements[-1]
                }
            team_table_rows.append(r)
    return team_table_rows


def process_decade_folder(folder_path):
    """
    Returns three lists: 
        `all_games_table_rows`, 
        `all_events_table_rows`, 
        `all_team_table_rows`
    """
    all_games_table_rows = []
    all_events_table_rows = []
    all_team_table_rows = []
    
    for f_name in os.listdir(folder_path):
        if f_name.endswith('.ROS'): # skip roster files
            continue
            
        f_path = os.path.join(folder_path, f_name)
        if f_name.startswith('TEAM'): # process team file
            team_table_rows = process_team_file(f_path)
            all_team_table_rows += team_table_rows
        else: # process events file
            games_table_rows, events_table_rows = process_event_file(f_path)
            all_games_table_rows += games_table_rows
            all_events_table_rows += events_table_rows
    
    return all_games_table_rows, all_events_table_rows, all_team_table_rows


def process_all_decades(data_path):
    """
    Returns all data from the given directory as three lists:
        `master_games_table_rows`, 
        `master_events_table_rows`, 
        `master_team_table_rows`
    """
    print('Processing...')
    master_games_table_rows = []
    master_events_table_rows = []
    master_team_table_rows = []
    
    for decade_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, decade_folder)
        games, events, team = process_decade_folder(folder_path)
        master_games_table_rows += games
        master_events_table_rows += events
        master_team_table_rows += team
        print('Completed decade:\t', decade_folder)
        
    return (
        master_games_table_rows, 
        master_events_table_rows, 
        master_team_table_rows
    )


def main():
	"""
	Scrape data and save csvs in the './data' folder.
	Delete file directories afterwards.
	If './data' does not exist, the files will be downloaded and saved first.
	"""
	data_path = './data'
	if 'data' not in os.listdir():
		start_year = input(f'''{"#"*40}
Press `enter`to download full dataset (beginning 1910).
Otherwise, choose from: 
	[ s ] (`small`: starting 1980)
	or
	[1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]
>>> ''')
		if not start_year:
			start_year = 1910
		elif start_year == 's':
			start_year = 1980
		elif start_year in [str(x) for x in range(1910, 2020, 10)]:
			start_year = int(start_year)
		else:
			print(f'\n\nCannot complete task with input: {start_year}')
			quit()
		print(f'Success! Collecting data from: {start_year}')
		create_data_directory(start_year)

	games, events, teams = process_all_decades(data_path)

	print('Creating DataFrames...')
	games_df = pd.DataFrame(games)
	events_df = pd.DataFrame(events)
	teams_df = pd.DataFrame(teams)

	games_df.to_csv(os.path.join(data_path, 'games.csv'))
	print(f'games.csv saved\tShape:{games_df.shape}')
	events_df.to_csv(os.path.join(data_path, 'events.csv'))
	print(f'events.csv saved\tShape:{events_df.shape}')
	teams_df.to_csv(os.path.join(data_path, 'teams.csv'))
	print(f'teams.csv saved\tShape:{teams_df.shape}')

	print('Deleting decades directories...')
	for f in os.listdir(data_path):
		if f.endswith('.csv'):
			continue
		shutil.rmtree(f'./data/{f}')
		print(f'Deleted:\t{f}')
	print('Complete.')



if __name__ == '__main__':
	main()