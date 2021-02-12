##################################################
# Support functions for baseball player project. #
##################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
import bz2
import warnings
from IPython.display import clear_output

from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.metrics import (classification_report, 
							 balanced_accuracy_score, 
							 log_loss)
import tensorflow as tf


class DataStorage:
	def __init__(self):
		"""Create DataStorage object for Dash app."""

		self.locked_in = []
		self.currently_trying = []


class Simulator:
	"""
	A Simulator class with the ability to simulate innings of baseball for a 
	given lineup against a given pitcher.
	These simulations can be used to "optimize" a batting order.
		*Find the order of hitters which yields the highest 
		`expected_runs_scored`*
	Results will vary since we are dealing with simulations. The larger the
	number of simulations run, the more robust the outcomes will be.

	Parameters:
	-----------
	model:
		A trained neural network that mas the method `.predict()` which returns 
		an array of probabilities.
	league:
		A populated league_player_classes.League object.
	X_preprocessor / y_preprocessor:
		Fit pipelines or preprocessors which will transform the data to make
		predictions on.
	player_finder:
		A PlayerFinder class to show the names of the players and the 
		association between player -> player_id.

	Methods:
	--------
	simulate():
		Begin a simulation of a given situation (lineup vs pitcher).
		Game data (temperature, field conditions) can also be input.
		The simulation will run a desired number of times for a given number 
			of innings per simulation.
		A report will be printed of the runs scored during ths simulation.
		The mean runs_scored will be returned.
	optimize_lineup():
		A given lineup (against a given pitcher) is shuffled methodically to
			discover the lineup order which is expected to return the highest
			runs-per-game.
		A printout will be shown during the simulations of the status.
		A list of player_ids will be returned in the "optimal" order.

	Example:
	--------
	>>> simulator = bsb.Simulator(model, MLB, X_preprocessor, y_preprocessor, pf)
	>>> simulator
	Simulator()

	>>> # Highest-scoring team of 2010
	>>> yanks_2010_opening_day_lineup = [
	>>>     ('Jeter', 'Derek'),
	>>>     ('Johnson', 'Nick'),
	>>>     ('Teixeira', 'Mark'),
	>>>     ('Rodriguez', 'Alex'),
	>>>     ('Cano', 'Robinson'),
	>>>     ('Posada', 'Jorge'),
	>>>     ('Granderson', 'Curtis'),
	>>>     ('Swisher', 'Nick'),
	>>>     ('Gardner', 'Brett')
	>>> ]
	>>>     
	>>> yanks_2010_ids = [simulator.player_finder.get_player_id(last_n, first_n) 
	>>>                   for last_n, first_n in yanks_2010_opening_day_lineup]
	>>> np.random.shuffle(yanks_2010_ids)
	>>> yanks_2010_ids
	['rodra001',
	 'gardb001',
	 'swisn001',
	 'teixm001',
	 'canor001',
	 'johnn001',
	 'granc001',
	 'posaj001',
	 'jeted001']

	>>> maddux = simulator.player_finder.get_player_id('Maddux', 'Greg')
	>>> maddux
	'maddg002'

	>>> opt_lineup = simulator.optimize_lineup(yanks_2010_ids, sheets, inning_half=1)
	********************************************************************************
	*                                Lineup locks:                                 *
	********************************************************************************
		1. Mark Teixeira
		2. Derek Jeter
		3. Nick Swisher
		4. Alex Rodriguez
		5. Jorge Posada
		6. Brett Gardner
		7. Robinson Cano
		8. Nick Johnson
		9. Curtis Granderson
	"""

	def __init__(self, 
				 model, 
				 league, 
				 X_preprocessor, 
				 y_preprocessor, 
				 player_finder):
		self.model = model
		self.league = league
		self.X_preprocessor = X_preprocessor
		self.y_preprocessor = y_preprocessor
		self.player_finder = player_finder

	def __repr__(self):
		return 'Simulator()'

	def __outcome_consequence(self, outcome):
		"""
		Returns a dictionary with results of the outcome.
		{outs, advances_runners, station-to-station}
		"""

		consequences = {
			'BB': {'results_in_out': 0, 'advances_runners': 1, 's2s': True},
			'D': {'results_in_out': 0, 'advances_runners': 2, 's2s': False},
			'E': {'results_in_out': 0, 'advances_runners': 1, 's2s': False},
			'HR': {'results_in_out': 0, 'advances_runners': 4, 's2s': False},
			'I': {'results_in_out': 0, 'advances_runners': 1, 's2s': True},
			'K': {'results_in_out': 1, 'advances_runners': 0, 's2s': False},
			'O': {'results_in_out': 1, 'advances_runners': 0, 's2s': False},
			'S': {'results_in_out': 0, 'advances_runners': 1, 's2s': False},
			'SAC': {'results_in_out': 1, 'advances_runners': 1, 's2s': False},
			'T': {'results_in_out': 0, 'advances_runners': 3, 's2s': False}
		}
		return consequences.get(outcome)

	def __advance_runners(self, simulation, o, n, s2s):
		"""
		Returns a new simulation based on bases advanced.
		Parameters:
		-----------
		`o`: outs, `n`: advances_runners, `s2s`: station-to-station.
		"""

		s = simulation.copy()

		# Advance batters according to `station-to-station`.
		if s2s:
			if 1 not in s['bases']: # if no one is on first.
				s['bases'] = np.append(s['bases'], 1) # batter>1
			elif (1 in s['bases']) & (2 not in s['bases']) & (3 in s['bases']):
				s['bases'] = np.append(s['bases'], 2) # batter>1, 1st>2nd, 3>3
			else:
				s['bases'] = np.append(s['bases'], 0) # set batter on base `0`
				s['bases'] += n
		else:
			if not o:
				s['bases'] = np.append(s['bases'], 0) # set batter on base `0`
			s['bases'] += n # everyone moves up n-amount

		for r in s['bases']: # count runs scored.
			if r >= 4:
				s['runs'] += 1
		s['bases'] = np.array([x for x in s['bases'] if 0 < x < 4])
		return s

	def __update_simulation(self, simulation, outcome):
		"""Returns a simulation updated from an outcome."""

		s = simulation.copy()
		consequence = self.__outcome_consequence(outcome)
		o = consequence.get('results_in_out')
		ar = consequence.get('advances_runners')
		s2s = consequence.get('s2s')

		# Update simulation.
		s['outs'] += o
		if s['outs'] >= 3:
			return s
		if ar: # advance runners.
			s = self.__advance_runners(s, o, ar, s2s)
		return s

	def __new_simulation(self):
		"""Blank representation of a baseball simulation."""

		return {
			'runs': 0,
			'outs': 0,
			'bases': np.array([])
		}

	def __describe_simulation_state(self, s):
		"""Prints the state of the simulation."""

		print(s['outs'], 'outs.')
		if 1 in s['bases'] and 2 in s['bases'] and 3 in s['bases']:
			print('Bases loaded.')
		else:
			onbase = 'first and second.' if (
						(1 in s['bases']) and (2 in s['bases'])) \
					 else 'first and third.' if (
						(1 in s['bases']) and (3 in s['bases'])) \
					 else 'second and third.' if (
						(2 in s['bases']) and (3 in s['bases'])) \
					 else 'first.' if 1 in s['bases'] \
					 else 'second.' if 2 in s['bases'] \
					 else 'third.' if 3 in s['bases'] \
					 else None
			if onbase:
				print('Man on', onbase)
			else:
				print('Nobody on.')
		print(s['runs'], 'scored so far this inning.')
		print('*'*30)

	def __make_df(self, 
				  hitter_id,
				  pitcher_id,
				  inning_num,
				  inning_half,
				  use_career_stats,
				  dblhdr_number,
				  day_night,
				  temp,
				  wind,
				  field_cond,
				  precip,
				  attendance,
				  previous_outcomes=None):
		"""Returns a single-row dataframe to perform predictions on."""

		if not use_career_stats:
			h_ab = 0
			h_k = -1
			h_ops = -1
			p_ip = 0.0
			p_whip = -1
			p_k_bb = -1
		else:
			(h_ab, 
			 h_k, 
			 h_ops) = self.league.get_player(
				hitter_id, 'hitter', create_new=True).stats_heading_in()
			(p_ip,
			 p_whip,
			 p_k_bb) = self.league.get_player(
				pitcher_id, 'pitcher', create_new=True).stats_heading_in()

		if previous_outcomes:
			if len(previous_outcomes) == 1:
				lag1, = previous_outcomes
				lag2 = lag3 = lag4 = lag5 = 'NONE'
			elif len(previous_outcomes) == 2:
				lag1, lag2 = previous_outcomes
				lag3 = lag4 = lag5 = 'NONE'
			elif len(previous_outcomes) == 3:
				lag1, lag2, lag3 = previous_outcomes
				lag4 = lag5 = 'NONE'
			elif len(previous_outcomes) == 4:
				lag1, lag2, lag3, lag4 = previous_outcomes
				lag5 = 'NONE'
			elif len(previous_outcomes) >= 5:
				lag1, lag2, lag3, lag4, lag5, = previous_outcomes[-5:]
		else:
			lag1 = lag2 = lag3 = lag4 = lag5 = 'NONE'

		row = {
			'inning_num': inning_num,
			'inning_half': inning_half,
			'hitter_id': hitter_id,
			'pitcher_id': pitcher_id,
			'h_ab_coming_in': h_ab,
			'h_k%_coming_in': h_k,
			'h_ops_coming_in': h_ops,
			'p_ip_coming_in': p_ip,
			'p_whip_coming_in': p_whip,
			'p_k_bb_coming_in': p_k_bb,
			'dblhdr_number': dblhdr_number,
			'day_night': day_night,
			'temp': temp,
			'wind': wind,
			'field_cond': field_cond,
			'precip': precip,
			'attendance': attendance,
			'prior_outcome_lag1': lag1,
			'prior_outcome_lag2': lag2,
			'prior_outcome_lag3': lag3,
			'prior_outcome_lag4': lag4,
			'prior_outcome_lag5': lag5
		}

		return pd.DataFrame([row])

	def __get_predictions_from_df(self, df):
		"""Returns a list of predictions for the lineup from self.model."""

		return self.model.predict_proba(
			self.X_preprocessor.transform(df)
		)[0]

	def simulate_branch(self, 
						lineup, 
						pitcher_id, 
						inning_num, 
						inning_half, 
						use_career_stats, 
						dblhdr_number,
						day_night,
						temp,
						wind,
						field_cond,
						precip,
						attendance,
						num_innings,
						verbose=True):
		"""
		Simulates a `branch` of a full simulation.
		A `branch` consists of a single count of {n} innings.
		"""

		verbose_dct = {
			'BB': 'a walk',
			'D': 'a double',
			'E': 'an error',
			'HR': 'a home run',
			'I': "defensive interferance",
			'K': 'a strikeout',
			'O': 'an out',
			'S': 'a single',
			'SAC': 'a sacrifice',
			'T': 'a triple',
			1: 'the first',
			2: 'the second',
			3: 'the third',
			4: 'the fourth',
			5: 'the fifth',
			6: 'the sixth',
			7: 'the seventh',
			8: 'the eighth',
			9: 'the ninth'
		}

		OUTCOMES = self.y_preprocessor.classes_
		results = []
		results_verbose = []

		atbat_idx = 0
		for inning in range(inning_num, inning_num+num_innings):
			# if verbose:
			top_bot = 'Bottom' if inning_half else 'Top'
			inning_statement = f"{top_bot} of {verbose_dct[inning]}."
			results_verbose.append(inning_statement)
			# print(headerize(inning_statement))

			previous_outcomes = []
			s = self.__new_simulation() # make new simulation
			while s['outs'] < 3:
				# generate df for hitter (lineup[atbat_idx])
				df = self.__make_df(
					lineup[atbat_idx],
					pitcher_id,
					inning_num,
					inning_half,
					use_career_stats,
					dblhdr_number,
					day_night,
					temp,
					wind,
					field_cond,
					precip,
					attendance,
					previous_outcomes=previous_outcomes
				)

				# get predictions
				predictions = self.__get_predictions_from_df(df)

				# select outcome
				simulated_outcome = np.random.choice(OUTCOMES, p=predictions)

				o1 = 'With'
				o2 = self.player_finder.get_player_name(lineup[atbat_idx])
				o3 = 'at bat, the play resulted in'
				o4 = verbose_dct[simulated_outcome]
				outcome_statement = f'{o1} **{o2}** {o3} **{o4}**.'
				# outcome_statement = 'With ' + \
				# 	self.player_finder.get_player_name(lineup[atbat_idx]) + \
				# 	' at bat, the play resulted in ' + \
				# 	 f'{verbose_dct[simulated_outcome]}.'
				results_verbose.append(outcome_statement)

				# update simulation
				s = self.__update_simulation(s, simulated_outcome)
				# if verbose:
					# self.__describe_simulation_state(s)

				# append `previous_outcomes`
				previous_outcomes.append(simulated_outcome)

				# advance lineup_idx and rollover if necessary
				atbat_idx += 1
				if atbat_idx >= len(lineup):
					atbat_idx = 0
				
				# append results
				if s.get('outs') >= 3:
					results.append(s.get('runs'))
		return results, results_verbose

	def simulate(self, 
				 lineup, 
				 pitcher_id, 
				 inning_num=1, 
				 inning_half=0, 
				 use_career_stats=True, 
				 dblhdr_number=0,
				 day_night='night',
				 temp=-1,
				 wind=-1,
				 field_cond='unknown',
				 precip='unknown',
				 attendance=-1,
				 n=10, 
				 num_innings=1,
				 verbose=0):
		"""
		Runs n-simulations based on model predictions.
		Returns `expected_runs_scored` after the simulations.

		**This simulation does not factor fielding or baserunning.**

		Parameters:
		-----------
		lineup: array or list of player_ids (strings)
			Ordered list of hitter_ids that represent the batting lineup.
		pitcher_id: string
			pitcher_id which represents the opposing pitcher.
		inning_num: int (default: 1)
			Inning number to start the simulation on.
		inning_half: int (one of [0, 1] :: default: 0)
			0: Top of the inning. 
			1: Bottom of the inning.
		use_career_stats: bool (default: True)
			If True, the player_finder will use career stats for the player.
			Otherwise, these fields will be left as `missing`.
		dblhdr_number: int (one of [0, 1, 2] :: default: 0)
			If `0`, the game is the only game scheduled on the day.
			If `1`, the game is the first of two scheduled for the same day.
			If `2`, the game is the second of two scheduled for the same day.
		day_night: string (one of ['day', 'night'] :: default: 'night')
			Time of day the game is played.
		temp: int (default: -1)
			Degrees Fahrenheit at the start of the game.
			`-1` represents an unknown value.
		wind: int (default: -1)
			Wind speed (mph) at the start of the game.
			`-1` represents an unknown value.
		field_cond: string (default: 'unknown')
			Recognized conditions are: 
				[`unknown`, `dry`, `wet`, `damp`, `soaked`]
		precip: string (default: 'unknown')
			Recognized percipitation values are: 
				[`unknown`, `none`, `rain`, `drizzle`, `showers`, `snow`]
		attendance: int (default: -1)
			Number of fans in attendance at the game.
			`-1` represents an unknown value.
		n: int (default: 10_000)
			Number of simulations to run.
		num_innings: int (default: 1)
			Number of innings to simulate per simulation.
		verbose: int (one of [`0`, `1`, `2`, `3`] :: default: 1)
			Level of verbosity while the simulator runs.
			* If `4`, every play of every simulation will be printed.
		"""

		RUNS_SCORED = []
		if verbose:
			print(f'Running {n} simulations of {num_innings} innings each.')

		# Run simulation.
		RESULTS = []
		for i in range(n):
			sim_results, verbose_results = self.simulate_branch(
				lineup=lineup, 
				pitcher_id=pitcher_id, 
				inning_num=inning_num, 
				inning_half=inning_half, 
				use_career_stats=use_career_stats, 
				dblhdr_number=dblhdr_number,
				day_night=day_night,
				temp=temp,
				wind=wind,
				field_cond=field_cond,
				precip=precip,
				attendance=attendance,
				num_innings=num_innings,
				verbose=False)

			RESULTS.append(sim_results)
			# try:
			# 	if i % (n//10) == 0:
			# 		if i > 0:
			# 			print(f'\tCompleted {i} branches...')
			# except ZeroDivisionError:
			# 	pass

		# Results.
		results_cols = [f'inning_{i}' 
						for i in range(inning_num, num_innings+1)]
		RESULTS = pd.DataFrame(
			RESULTS, 
			columns=results_cols, 
			index=pd.Index(range(1, n+1), name='game')
			)
		runs_per_inning = RESULTS.mean().mean()
		RESULTS['simulation_total'] = RESULTS.sum(axis=1)
		verbose_results = f'- Average Runs Scored (/inning): {round(runs_per_inning, 3)}\n- Average Runs Scored (/9-inning-game): {round((runs_per_inning)*9, 3)}'
		if verbose:
			print(headerize('Simulation Complete'))
			print(f'{n} {num_innings}-inning simulations run.')
			print(
				'Average Runs Scored (/inning):',
				round(runs_per_inning, 3)
				)
			print(
				'Average Runs Scored (/9-inning-game):',
				round((runs_per_inning)*9, 3)
				)
			with plt.style.context(['ggplot', 'seaborn-talk']):
				fig, ax = plt.subplots(figsize=(6,4))
				sns.distplot(RESULTS['simulation_total'])
				ax.set(
					title='Runs Scored Per Game',
					xlabel='Runs Scored',
					ylabel='Percent of Simulated Innings'
					)
				plt.show()
		return RESULTS, verbose_results
	
	def optimize_lineup(self,
						hitters_lst, 
						pitcher_id, 
						inning_num=1, 
						inning_half=0, 
						use_career_stats=True, 
						dblhdr_number=0,
						day_night='night',
						temp=-1,
						wind=-1,
						field_cond='unknown',
						precip='unknown',
						attendance=-1,
						simulations_per_order=150):
		"""
		Iterates through each spot in the lineup finding the highest
		`expected_runs_scored`.

		(e.g.:
			# try1: [(1),2,3,4,5,6,7,8,9]
			# try2: [2,(1),3,4,5,6,7,8,9]
			# try3: [3,2,(1),4,5,6,7,8,9]
			# try4: [4,2,3,(1),5,6,7,8,9]
			# try5: . . .
		)

		Once the highest `expected_runs_scored` is found from the first slot, 
		the first slot is "locked-in" and the second slot is tried.
		* Median is used instead of mean to be more robust to outliers. 
			(eg: a simulation where 10 runs were scored in an inning 
			 will dramatically skew the output.)

		Returns an optimal order based on `expected_runs_scored`.
		"""

		def show_lineup_locks(locked_in):
			"""Prints the lineup locks from a given list."""

			print(headerize('Lineup locks:'))
			for n, hitter in enumerate(locked_in, 1):
				name = self.player_finder.get_player_name(hitter, verbose=False)
				print(f'\t{n}. {name}')

		hitters = hitters_lst.copy()
		locked_in = []

		# Find optimized hitter for each spot in order.
		for start_idx in range(9): # finding the top players for first 8 spots.
			_hitters = [] # tuple of stats of the simulations
			for batting_order in shuffle_lst(hitters, 
											 start_idx, 
											 masked_elements=8-len(locked_in)):
				clear_output(wait=True)
				show_lineup_locks(locked_in)
				print('Trying:\n', np.array(
					[f'{n}. {self.player_finder.get_player_name(x, verbose=False)}'
					 for n, x in enumerate(batting_order, 1)]).reshape(3,3))

				# Run simulations.
				df, _ = self.simulate(
					lineup=batting_order, 
					pitcher_id=pitcher_id, 
					inning_num=inning_num,
					inning_half=inning_half,
					use_career_stats=use_career_stats,
					dblhdr_number=dblhdr_number,
					day_night=day_night,
					temp=temp,
					wind=wind,
					field_cond=field_cond,
					precip=precip,
					attendance=attendance,
					n=simulations_per_order, 
					num_innings=(start_idx//3)+1,
					verbose=False)

				# Append (hitter_id, total_runs, count_of_scores).
				runs = df['simulation_total'].copy()
				# Remove bottom and top 25%
				runs = runs[(runs > runs.quantile(0.25)) & 
							(runs < runs.quantile(0.75))]
				_scoring_sims = runs[runs > 0]
				_hitters.append(
					(batting_order[start_idx], 
					 runs.sum(), # sum of all the sims
					 len(_scoring_sims)) # number of sims with runs scored
				)

			# Find player with highest expected_runs_scored.
			top_hitter = sorted(
				_hitters, 
				key=lambda x: (x[1], x[2]), 
				reverse=True
				)[0][0]
			locked_in.append(top_hitter)

			# Remove locked in players from hitters list.
			hitters = [h for h in hitters if h not in locked_in]

			# Reset hitters list starting with locked in players 
			# so locked_in players won't be iterated over.
			hitters = locked_in + hitters

		# Display
		clear_output(wait=True)
		show_lineup_locks(locked_in)
		return hitters


def shuffle_lst(lst, start_idx=0, masked_elements=0, mask=''):
	"""
	Returns shuffles of a list where one element traverses the list indices.
	Can change `start_idx` so that the start of the list remains intact
	while `lst[start_idx:]` gets iterated through.

	If `masked_elements`, the final n elements of the list will be masked with
	the given mask value.

	e.g.:
	>>> shuffle_lst([1, 2, 3, 4])
	[[0, 1, 2, 3],
	 [1, 0, 2, 3],
	 [2, 1, 0, 3],
	 [3, 1, 2, 0]]
	 
	>>> shuffle_lst([1, 2, 3, 4], 1)
	[[0, 1, 2, 3],
	 [0, 2, 1, 3],
	 [0, 3, 2, 1]]

	>>> shuffle_lst(['a','b','c','d'], masked_elements=2, mask='FOO')
	[['a', 'b', 'FOO', 'FOO'],
	 ['b', 'a', 'FOO', 'FOO'],
	 ['c', 'b', 'FOO', 'FOO'],
	 ['d', 'b', 'FOO', 'FOO']]
	"""

	if start_idx >= len(lst):
		return lst

	sliced_lst = lst[start_idx:].copy()
	results = []
	for i in range(len(sliced_lst)):
		new = sliced_lst.copy()
		new[0], new[i] = new[i], new[0]
		if masked_elements:
			new[-masked_elements:] = [mask for _ in new[-masked_elements:]]
		results.append(lst[:start_idx] + new)
	return results


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

	def show_player(self, last_name, first_name=''):
		"""
		Print player records who match the given name (string).

		If a name is not found, suggestions are offered 
		for names close alphabetically.
		"""

		last_name = last_name.strip()

		# First, check if the name matches spelling of names in the database.
		if not first_name:
			results = self.player_df[
				self.player_df['last'].map(lambda x:
										   x.lower()) == last_name.lower()]
		else:
			results = self.player_df[
				(self.player_df['last'].map(
					lambda x: x.lower()) == last_name.lower()) & 
				(self.player_df['first'].map(
					lambda x: x.lower()) == first_name.lower())]

		if len(results) > 0:
			print(results)
			return

		# Capital first letter, but rest can be upper/lower like 'McAllister'.
		last_name = last_name[0].upper() + last_name[1:]
		print('No matches found.\nDid you mean one of:')
		print(self.__like_players_lst(last_name))
		return

	def get_player_name(self, player_id, verbose=False):
		"""
		Returns a string '{first_name} {last_name}' given a player_id.
		"""

		player_id = player_id.strip()
		try:
			player = self.player_df.loc[player_id]
			return f"{player['first']} {player['last']}"
		except KeyError:
			if verbose:
				print(f'No player found for id: {player_id}.')
				print('Try running `.show_player()` to get `player_id`.')
			else:
				return 'None'

	def get_player_id(self, last_name, first_name, play_debut_year=1920):
		"""
		Returns the player_id for a given name.
		If more than one players share the name, you can pass a debut year 
		to further filer.
		If more than one entry is found still, all `player_ids` will be returned.
		"""

		matching_ids = list(self.player_df[
			(self.player_df['last'].map(
				lambda x: x.lower()) == last_name.lower()) &
			(self.player_df['first'].map(
				lambda x: x.lower()) == first_name.lower()) &
			(self.player_df['play_debut'] >= str(play_debut_year))
		].index.values)
		if len(matching_ids) == 0:
			return ''
		if len(matching_ids) == 1:
			return matching_ids[0]
		else:
			return matching_ids


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
		print('`./data` directory not found.')
		return False

	files = ['games.csv', 'events.csv', 'teams.csv']
	for f in files:
		if f not in os.listdir('./data'):
			print(f'`{f}` not found.')
			return False
	print(headerize('SUCCESS - Data Found'))
	return True


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


def plot_stat_impact_on_outcome(
	df, 
	stat_column, 
	legend_label=None,
	align='bottom',
	target='outcome', 
	bins=5, 
	labels=['low',
			'mid-low', 
			'mid', 
			'mid-high',
			'high']):
	"""
	Show and return a stacked barplot of a binned stat (via pd.qcut)
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
	align: string (default: 'bottom')
		One of ('bottom', 'center'). Aligns the bars either at the center point 
		of the "mid"-bin or aligns all bars to the left wall.
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
	bottom_align = align == 'bottom'

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
	x_tick_markers = list(range(len(possible_outcomes)))
	gradient_colors = ['firebrick',
					   'lightcoral',
					   'thistle',
					   'cornflowerblue',
					   'darkblue']

	# Plot
	fig, ax = plt.subplots(figsize=(12,8))
	for outcome, x_tick in zip(possible_outcomes, x_tick_markers):
		s = data[data[target] == outcome].copy()
		# Find leftmost placement of bars.
		if bottom_align:
			leftmost = 0
		else:
			leftmost = (s[s[stat_column] == 'mid']['percent'].values[0] / 2) + \
						s[s[stat_column] == 'mid-low']['percent'].values[0] + \
						s[s[stat_column] == 'low']['percent'].values[0]
			leftmost = -leftmost

		# Iterate and plot.
		for group, color in zip(labels, gradient_colors):
			if x_tick == x_tick_markers[0]:
				if legend_label:
					label = f'{group.title()} {legend_label}'
				else:
					label = group.title()
			else:
				label = None
			v = s[s[stat_column] == group]['percent'].values[0]
			ax.bar(x_tick, 
					v, 
					bottom=leftmost,
					color=color,
					label=label)
			leftmost += v
	ax.set_xticks(x_tick_markers)
	ax.set_xticklabels(possible_outcomes)
	if bottom_align:
		ax.set_ylim(0, 1)
		ax.set_yticklabels([f'{round(x*100)}%' for x in ax.get_yticks()])
	else:
		ax.set_yticks([])
	ax.set(title=f'{legend_label or stat_column} :: Impact on {target.title()}',
		   ylabel='Percent of Outcomes' if bottom_align else '',
		   xlabel='Outcome')
	handles, labels = ax.get_legend_handles_labels() # In order to reverse the legend elements.
	ax.legend(
		handles[::-1], 
		labels[::-1],
		loc='right', 
		bbox_to_anchor=(0,0,1.22,1),
		labelspacing=3,
		fancybox=True,
		ncol=1)
	fig.tight_layout()
	plt.show()
	return fig

def print_metrics(X, 
				  y, 
				  classifier, 
				  target_names=None, 
				  heading='', 
				  return_loss=False):
	"""
	Print the following metrics:
		- Classification Report
		- Balanced Accuracy
		- Log-Loss

	If `return_loss`, cross-entropy loss is returned.

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
	if isinstance(classifier, tf.python.keras.engine.sequential.Sequential):
		y_pred_proba = classifier.predict(X)
		y_pred = [np.argmax(x) for x in y_pred_proba]
	else:
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
	if return_loss:
		return loss

def load_modeling_tools(path='./modeling/'):
	"""
	Returns the following pickled objects from the given path:
		X_preprocessor (fit ColumnTransformer)
		y_preprocessor (fit LabelEncoder)
		WEIGHTS_DCT (dictionary)
		X_train_processed (sparse matrix)
		y_train_processed (sparse matrix)
		X_test_processed (sparse matrix)
		y_test_processed (sparse matrix)
		X_val_processed (sparse matrix)
		y_val_processed (sparse matrix)
	"""

	modeling_tools = {}
	print('Loading...')
	for f_dir in [x for x in os.listdir(path) if 'ipynb' not in x]:
		for fname in os.listdir(path+f_dir):
			name = fname.split('.')[0]
			if not name:
				continue
			if fname.endswith('.npz'):
				modeling_tools[name] = sparse.load_npz(f'{path}{f_dir}/{fname}')
			elif fname.endswith('.pkl'):
				with open(f'{path}{f_dir}/{fname}', 'rb') as infile:
					f = pickle.load(infile)
					modeling_tools[name] = f
			print('Loaded:', name)
	print('Complete!')
	return (
		modeling_tools.get('X_preprocessor'),
		modeling_tools.get('y_preprocessor'),
		modeling_tools.get('WEIGHTS_DCT'),
		modeling_tools.get('X_train_processed'),
		np.ravel(modeling_tools.get('y_train_processed').todense()),
		modeling_tools.get('X_test_processed'),
		np.ravel(modeling_tools.get('y_test_processed').todense()),
		modeling_tools.get('X_val_processed'),
		np.ravel(modeling_tools.get('y_val_processed').todense())
	)

def load_preprocessors(path='./modeling/preprocessor/'):
	"""
	Returns:
		X_preprocessor (fit ColumnTransformer)
		y_preprocessor (fit LabelEncoder)
	"""
	
	with open(f'{path}X_preprocessor.pkl', 'rb') as infile:
		X_preprocessor = pickle.load(infile)
	with open(f'{path}y_preprocessor.pkl', 'rb') as infile:
		y_preprocessor = pickle.load(infile)
	return X_preprocessor, y_preprocessor

def plot_history(history, style=['ggplot', 'seaborn-talk']):
	"""
	Plot history from History object (or history dict) 
	once Tensorflow model is trained.

	Parameters:
	-----------
	history:
		History object returned from a model.fit()
	style: string or list of strings (default: ['ggplot', 'seaborn-talk'])
		Style from matplotlib.
	"""
	if not isinstance(history, dict):
		history = history.history

	metrics_lst = [m for m in history.keys() if not m.startswith('val')]
	N = len(metrics_lst)
	with plt.style.context(style):
		fig, ax_lst = plt.subplots(nrows=N, figsize=(8, 4*(N)))
		ax_lst = [ax_lst] if N == 1 else ax_lst.flatten() # Flatten ax_lst.
		for metric, ax in zip(metrics_lst, ax_lst):
			val_m = f'val_{metric}'
			ax.plot(history[metric], label=metric)
			ax.plot(history[val_m], label=val_m)
			ax.set(title=metric.title(), xlabel='Epoch', ylabel=metric.title())
			ax.legend()
		fig.tight_layout()
		plt.show()