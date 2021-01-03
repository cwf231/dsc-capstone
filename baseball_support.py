##################################################
# Support functions for baseball player project. #
##################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import pickle
import bz2
import warnings

from sklearn.metrics import (classification_report, 
                             balanced_accuracy_score, 
                             log_loss)


class PlayerFinder:
	"""
	A PlayerFinder class which stores player data:
		[id, last, first, play_debut, mgr_debut, coach_debut, ump_debut]

	This class will be able to search for players (and importantly get their
	`player_id`) based on their last name.

	Example usage:
	--------------
	>>> playerfinder = PlayerFinder()
	>>> playerfinder.show_player('Griffey')
	             last first play_debut mgr_debut coach_debut ump_debut
	id                                                                
	grifk001  Griffey   Ken 1973-08-25       NaN  04/06/1993       NaN
	grifk002  Griffey   Ken 1989-04-03       NaN         NaN       NaN

	>>> playerfinder.get_player_name('jeted001')
	'Derek Jeter'
	"""
	def __init__(self, player_csv_path='players.csv'):
		"""
		Create class with players dataframe loaded.
			['id', 'last', 'first', 'play_debut', 
			'mgr_debut', 'coach_debut', 'ump_debut']
		
		Can search for players by name and get player_ids.
		"""
		self.player_df = self.__load_data(player_csv_path)

	def __repr__(self):
		return f'''
	**********PLAYER FINDER**********
	Number of player records loaded: {len(self.player_df)}'''
	    
	def __load_data(self, player_csv_path):
		"""Load and clean player data."""
		players = pd.read_csv(player_csv_path, index_col=0)
		players['play_debut'] = pd.to_datetime(players['play_debut'])
		players['first'] = players['first'].fillna('FNU')
		return players

	def __like_players_lst(self, last_name):
		"""Offer suggestions of similar names to the given input."""
		last_names_lst = sorted(self.player_df['last'].unique())

		while True:
			if len(last_names_lst) < 25:
				break
			mid = len(last_names_lst) // 2
			if last_name < last_names_lst[mid]:
				last_names_lst = last_names_lst[:mid]
			elif last_name > last_names_lst[mid]:
				last_names_lst = last_names_lst[mid:]
			else:
				mid -= 1
		return np.array(last_names_lst)

	def show_player(self, last_name):
		"""
		Print player records who match the given last name (string).

		If a name is not found, suggestions are offered 
		for names close alphabetically.
		"""
		last_name = last_name.strip()

		# First, check if the name matches spelling of names in the database.
		results = self.player_df[
			self.player_df['last'].map(lambda x:
									   x.lower()) == last_name.lower()]
		if len(results) > 0:
			print(results)
			return

		# Capital first letter, but rest can be upper/lower like 'McAllister'.
		last_name = last_name[0].upper() + last_name[1:]
		print('No matches found.\nDid you mean one of:')
		print(self.__like_players_lst(last_name))
		return

	def get_player_name(self, player_id):
		"""
		Returns a string '{first_name} {last_name}' given a player_id.
		"""
		player_id = player_id.strip()
		try:
			player = self.player_df.loc[player_id]
			return f"{player['first']} {player['last']}"
		except KeyError:
			print(f'No player found for id: {player_id}.')
			print('Try running `.show_player()` to get `player_id`.')


def underline(string, character='-'):
    """
    Return a string of a given character with the length of a given string.
    """
    return character * len(string)
    
    
def headerize(string, character='*', max_len=80):
    """
    Return a given string with a box (of given character) around it.
    """
    if max_len:
        # Create uniform size boxes for headers with centered text.
        if len(string) > max_len-2:
            string = string[:max_len-5] + '...'
            
        total_space = max_len - 2 - len(string)
        left = total_space // 2
        if total_space % 2 == 0:
            right = left
        else:
            right = left + 1
        
        top = character * max_len
        mid = f'{character}{" " * left}{string}{" " * right}{character}'
        bot = top
    else:
        # Create modular header boxes depending on the length of the string.
        top = character * (len(f'{string}')+42)
        mid = f'{character}{" " * 20}{string}{" " * 20}{character}'
        bot = top
        
    return f'{top}\n{mid}\n{bot}'


def check_for_data(path='./data'):
	"""
	Checks path for files: games.csv, events.csv, teams.csv.
	Also checks root directory for players.csv.
	"""
	if 'data' not in os.listdir():
		print('''`./data` directory not found. 
Please follow instructions to run `scrape_data_to_csv.py`''')
		return

	files = ['games.csv', 'events.csv', 'teams.csv']
	for f in files:
		if f not in os.listdir('./data'):
			print(f'`{f}` not found.')
			return
	print(headerize('SUCCESS - Data Found'))
	return


def load_data():
	"""Returns pandas dataframes: (`games`, `events`, `teams`)"""
	with warnings.catch_warnings():
		warnings.simplefilter(action='ignore', category=FutureWarning)
		games = pd.read_csv('./data/games.csv', index_col=0, low_memory=False)
		events = pd.read_csv('./data/events.csv', index_col=0, low_memory=False)
		teams = pd.read_csv('./data/teams.csv', index_col=0, low_memory=False)

	return games, events, teams


def engineer_outcome(series):
	"""
	Decipher events made by the batter at the plate.
	Information on this column can be found here: 
	    https://www.retrosheet.org/eventfile.htm
	    
	Returns a dataframe of two columns: [`outcome`, `total_bases`]
	"""
	#########################
	# DEFINE OUTCOME TYPES  #
	#########################
	# Hits.
	is_single = lambda x: re.match('S', x)
	is_double = lambda x: re.match('D', x)
	is_triple = lambda x: re.match('T', x)
	is_homerun = lambda x: re.match('H[^P]', x)

	# Drawn walk.
	is_walk = lambda x: re.match('W|I|IW|HP', x)

	# Strikeout.
	is_strikeout = lambda x: re.match('K', x)

	# Sacrifice.
	is_sacrifice = lambda x: re.search('\/SF|\/SH', x)

	# Out.
	is_groundout = lambda x: re.match('^[\d]{2}', x)
	is_flyout = lambda x: re.match('^[\d][^\d]|^[\d]$', x)
	is_fielders_choice = lambda x: re.match('FC', x)

	# Defensive failure.
	is_error = lambda x: re.match('E|FLE', x)
	is_interference = lambda x: re.match('C', x)

	# No result - baserunner stuff.
	is_no_play = lambda x: re.match('NP', x)
	is_no_result = lambda x: re.match('BK|CS|DI|OA|PB|WP|PO|SB', x)

	############################################################
	# Iterate through series and convert to simplified outcome.#
	############################################################
	data = []
	for outcome in series:
		if is_single(outcome):
			data.append({'outcome': 'S', 'total_bases': 1})
		elif is_double(outcome):
			data.append({'outcome': 'D', 'total_bases': 2})
		elif is_triple(outcome):
			data.append({'outcome': 'T', 'total_bases': 3})
		elif is_homerun(outcome):
			data.append({'outcome': 'HR', 'total_bases': 4})
		elif is_walk(outcome):
			data.append({'outcome': 'BB', 'total_bases': 0})
		elif is_strikeout(outcome):
			data.append({'outcome': 'K', 'total_bases': 0})
		elif is_sacrifice(outcome):
			data.append({'outcome': 'SAC', 'total_bases': 0})
		elif is_groundout(outcome) or \
			 is_flyout(outcome) or \
			 is_fielders_choice(outcome):
			data.append({'outcome': 'O', 'total_bases': 0})
		elif is_error(outcome):
			data.append({'outcome': 'E', 'total_bases': 0})
		elif is_interference(outcome):
			data.append({'outcome': 'I', 'total_bases': 0})
		elif is_no_play(outcome):
			data.append({'outcome': np.nan, 'total_bases': np.nan})
		elif is_no_result(outcome):
			data.append({'outcome': np.nan, 'total_bases': np.nan})
            
	return pd.DataFrame(data)


def plot_stat_impact_on_outcome(df, 
								stat_column, 
								legend_label=None,
								target='outcome', 
								bins=5, 
								labels=['low',
										'mid-low', 
										'mid', 
										'mid-high',
										'high']):
	"""
	Show a stacked barplot of a binned stat (via pd.qcut)
	and its relationship with the target.

	Parameters:
	-----------
	df: Pandas DataFrame
	stat_column: string
		Column to bin and plot against the target.
	legend_label: string or None (default: None)
		Pass a suffix for the legend. This should be a shorter abbreviation
		of the `stat_column`.
		If None is passed, there will be no suffix to the labels.
	target: string (default: 'outcome')
		Categorical column to plot.
	bins: int (default: 5)
		Number of bins to fit the `stat_column` into.
	labels: list or False (default: [`low`, . . . , `high`])
		Names of labels (from lowest to highest).
	"""
	data = (
		pd.concat(
			[pd.qcut(df[stat_column], bins, labels=labels),
			df[target]], 
			axis=1)
		.groupby(target)[stat_column]
		.value_counts(normalize=True)
		.rename('percent')
		.reset_index()
	)

	# Get sorted outcomes.
	if len(labels) % 2 == 0:
		top_half_index = len(labels)//2
	else:
		top_half_index = (len(labels)//2) + 1

	possible_outcomes = (
		data[data[stat_column].isin(labels[top_half_index:])]
			.groupby(target)
			.sum()
			.sort_values('percent', ascending=False)
			.index
			.values
		)

	# Housekeeping for plot.
	y_tick_markers = list(range(len(possible_outcomes), -1, -1))
	gradient_colors = ['firebrick',
					   'lightcoral',
					   'gainsboro',
					   'cornflowerblue',
					   'darkblue']

	# Plot
	fig, ax = plt.subplots()
	for outcome, y_tick in zip(possible_outcomes, y_tick_markers):
		s = data[data[target] == outcome].copy()
		# Find leftmost placement of bars.
		leftmost = (s[s[stat_column] == 'mid']['percent'].values[0] / 2) + \
					s[s[stat_column] == 'mid-low']['percent'].values[0] + \
					s[s[stat_column] == 'low']['percent'].values[0]
		leftmost = -leftmost

		# Iterate and plot.
		for group, color in zip(labels, gradient_colors):
			if y_tick == y_tick_markers[0]:
				if legend_label:
					label = f'{group.title()} {legend_label}'
				else:
					label = group.title()
			else:
				label = None
			v = s[s[stat_column] == group]['percent'].values[0]
			ax.barh(y_tick, 
					v, 
					left=leftmost,
					color=color,
					label=label)
			leftmost += v
	ax.set_yticks(y_tick_markers)
	ax.set_yticklabels(possible_outcomes)
	ax.set(title=f'{stat_column} :: Impact on {target.title()}',
		   xlabel='Percent of Outcomes',
		   ylabel='Outcome')
	ax.legend(loc='upper center', 
			  bbox_to_anchor=(0.5, -0.15),
			  fancybox=True, 
			  shadow=True, 
			  ncol=5)
	fig.tight_layout()
	plt.show()

def print_metrics(X, y, classifier, target_names=None, heading=''):
	"""
	Print the following metrics:
		- Classification Report
		- Balanced Accuracy
		- Log-Loss

	Parameters:
	-----------
	X:
		Array with predictive data to show report on.
	y:
		Array with the target data to show report on.
	classifier:
		Fit classifier with method `self.predict_proba()`
	target_names:
		List of target names for the classification report.
	heading:
		String to print at the top of the report.
	"""
	# Get predictions.
	y_pred = classifier.predict(X)
	y_pred_proba = classifier.predict_proba(X)

	# Get metrics.
	clf_report = classification_report(y, y_pred, 
									   target_names=target_names)
	bal_acc = balanced_accuracy_score(y, y_pred)
	loss = log_loss(y, y_pred_proba)

	# Printout.
	if heading:
		print(headerize(heading))
	print(clf_report)
	print('            Balanced Accuracy:', bal_acc)
	print('Cross-Entropy Loss (Log-Loss):', loss)

def load_modeling_tools(path='./modeling/'):
	"""
	Returns the following pickled objects from the given path:
		X_preprocessor (fit ColumnTransformer)
		y_preprocessor (fit LabelEncoder)
		X_train_processed (sparse matrix)
		y_train_processed (numpy array)
		X_test_processed (sparse matrix)
		y_test_processed (numpy array)
	"""
	modeling_tools = {}
	print('Loading...')
	for fname in os.listdir(path):
		with bz2.open(f'{path}{fname}', 'rb') as infile:
			name = fname.split('.')[0]
			f = pickle.load(infile)
			modeling_tools[name] = f
			print('Loaded:', name)
	print('Complete!')
	return (
    	modeling_tools.get('X_preprocessor'),
		modeling_tools.get('y_preprocessor'),
		modeling_tools.get('X_train_processed'),
		modeling_tools.get('y_train_processed'),
		modeling_tools.get('X_test_processed'),
		modeling_tools.get('y_test_processed'),
	)
