################################################################################
# This file contains classes for the League and Players.					   #
# This will be useful to store individual time series stats (event-to-event)   #
# as well as adding `{stat}_coming_in` to a dataframe.						   #
################################################################################
# Example usage:
# --------------
# >>> mlb = League()
# >>> mlb
# Number of Hitters: 0
# Number of Pitchers: 0
# Total Players: 0

# >>> claude = Hitter('cwf231')
# >>> claude
# Hitter(cwf231)

# >>> mlb.add_player(claude)
# >>> mlb
# Number of Hitters: 1
# Number of Pitchers: 0
# Total Players: 1

# >>> mlb.get_player('cwf231')
# Hitter(cwf231)

# >>> claude.stats_heading_in()
# (0, -1.0, -1.0)

# >>> claude.process_event_outcome('HR', 4)
# >>> claude.process_event_outcome('S', 1)
# >>> claude.process_event_outcome('BB', 0)
# >>> claude.process_event_outcome('K', 0)
# >>> claude.stats_heading_in()
# (3, 0.3333333333333333, 2.416666666666667)

# >>> claude.rolling_stats
# [{'bb': 0, 'k': 0, 'h': 0, 'sac': 0, 'tb': 0, 'ab': 0},
#  {'bb': 0, 'k': 0, 'h': 1, 'sac': 0, 'tb': 4, 'ab': 1},
#  {'bb': 0, 'k': 0, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 2},
#  {'bb': 1, 'k': 0, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 2},
#  {'bb': 1, 'k': 1, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 3}]
################################################################################
import pandas as pd


class League:
	"""
	A representation of a league (MLB).
	This class houses Players (Hitters and Pitchers).

	Example usage:
	--------------
	>>> mlb = League()
	>>> mlb
	Number of Hitters: 0
	Number of Pitchers: 0
	Total Players: 0

	>>> claude = Hitter('cwf231')
	>>> claude
	Hitter(cwf231)

	>>> mlb.add_player(claude)
	>>> mlb
	Number of Hitters: 1
	Number of Pitchers: 0
	Total Players: 1

	>>> mlb.get_player('cwf231')
	Hitter(cwf231)
	"""

	def __init__(self):
		"""Creates an empty League."""

		self.hitters = {}
		self.pitchers = {}

	def __repr__(self):
		return f'''
	**********LEAGUE**********
	Number of Hitters:  {len(self.hitters)}
	Number of Pitchers: {len(self.pitchers)}
	Total Players:      {len(self.hitters) + len(self.pitchers)}'''

	def add_player(self, player):
		"""Adds a Player to it's storage."""

		if isinstance(player, Hitter):
			if player.player_id in self.hitters:
				return
			self.hitters[player.player_id] = player
		elif isinstance(player, Pitcher):
			if player.player_id in self.pitchers:
				return
			self.pitchers[player.player_id] = player

	def get_player(self, player_id, hitter_pitcher='', create_new=False):
		"""
		Returns a player (or None) from the League.
		
		If `hitter_pitcher` is not passed, both will be searched for.
		If both player-records are found, (hitter, pitcher) will be returned.
		If one player-record is found, the player will be returned normally.

		`hitter_pitcher` can be one of ('hitter', 'pitcher').
		If `create_new` is True and a record has not been found, a new player
		will be created and added to the league.
			`create_new` can only be run if `hitter_pitcher` is passed.
		"""

		if hitter_pitcher:
			if hitter_pitcher not in ('hitter', 'pitcher'):
				raise Exception("hitter_pitcher must be ('hitter' | 'pitcher')")
			if hitter_pitcher == 'hitter':
				player =  self.hitters.get(player_id, None)
			elif hitter_pitcher == 'pitcher':
				player =  self.pitchers.get(player_id, None)

			if not player and create_new:
				if hitter_pitcher == 'hitter':
					player = Hitter(player_id)
				else:
					player = Pitcher(player_id)
				self.add_player(player)
		else:
			player_h = self.hitters.get(player_id, None)
			player_p = self.pitchers.get(player_id, None)
			if player_h and player_p:
				return player_h, player_p
			elif player_h:
				return player_h
			elif player_p:
				return player_p

		return player

	def populate_league_with(self, events, update_dataframe=True):
		"""
		Takes a dataframe (`events`) with columns:
			['game_id', 'inning_num', 'inning_half', 'hitter_id', 
			 'pitcher_id', 'outcome', 'total_bases', 'date']
		Iterates through each row and creates / updates each player's
		career- and rolling-stats.

		Populates the League with all of the Players and Player-stats.

		If `update_dataframe`, 
		adds columns to the passed `events` dataframe:
			['h_ab_coming_in', 'h_k%_coming_in', 'h_ops_coming_in',
			 'p_ip_coming_in', 'p_whip_coming_in', 'p_k_bb_coming_in']
		"""

		ab_lst = []
		k_perc_lst = []
		ops_lst = []
		ip_lst = []
		whip_lst = []
		k_bb_lst = []

		for idx, info in events.iterrows():
			# Get or set Hitter & Pitcher.
			h = self.get_player(info['hitter_id'], 'hitter', create_new=True)
			p = self.get_player(info['pitcher_id'], 'pitcher', create_new=True)

			# Set hitter & pitcher `stats_coming_in`.
			ab, k_perc, ops = h.stats_heading_in()
			ip, whip, k_bb = p.stats_heading_in()
			ab_lst.append(ab)
			k_perc_lst.append(k_perc)
			ops_lst.append(ops)
			ip_lst.append(ip)
			whip_lst.append(whip)
			k_bb_lst.append(k_bb)

			# Update hitter & pitcher stats based on event outcome.
			h.process_event_outcome(info['outcome'], info['total_bases'])
			p.process_event_outcome(info['outcome'])

		events['h_ab_coming_in'] = ab_lst
		events['h_k%_coming_in'] = k_perc_lst
		events['h_ops_coming_in'] = ops_lst
		events['p_ip_coming_in'] = ip_lst
		events['p_whip_coming_in'] = whip_lst
		events['p_k_bb_coming_in'] = k_bb_lst


class Player:
	"""
	A Baseball Player class.
	The class will store the player's stats from event-to-event as well as 
	the player's `career_stats` (the most up-to-date stats for the player).

	A `Player` is either `Hitter` or a `Pitcher`.

	Note: Pitcher stats are more harsh than normal because he is being judged
	merely on the interaction with the hitter and only the hitter.
	For example:
	    - Outs made by runners caught stealing are not counted towards 
	    a Pitcher's outs.
	    - Double Plays are only counted as one out for the Pitcher.

	Example usage:
	--------------
	>>> claude = Hitter('cwf231')
	>>> claude
	Hitter(cwf231)

	>>> claude.stats_heading_in()
	(0, -1.0, -1.0)

	>>> claude.process_event_outcome('HR', 4)
	>>> claude.process_event_outcome('S', 1)
	>>> claude.process_event_outcome('BB', 0)
	>>> claude.process_event_outcome('K', 0)
	>>> claude.stats_heading_in()
	(3, 0.3333333333333333, 2.416666666666667)

	>>> claude.rolling_stats
	[{'bb': 0, 'k': 0, 'h': 0, 'sac': 0, 'tb': 0, 'ab': 0},
	 {'bb': 0, 'k': 0, 'h': 1, 'sac': 0, 'tb': 4, 'ab': 1},
	 {'bb': 0, 'k': 0, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 2},
	 {'bb': 1, 'k': 0, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 2},
	 {'bb': 1, 'k': 1, 'h': 2, 'sac': 0, 'tb': 5, 'ab': 3}]
	"""

	def __init__(self, player_id):
		"""
		Creates the Player. 

		Attributes:
			player_id
			career_stats
			rolling_stats
		"""

		self.player_id = player_id
		self.career_stats = {}
		self.rolling_stats = []
	    
	def __repr__(self):
		return f'Player({self.player_id})'

	def __str__(self):
		return f"""{self.player_id}
Career stats:\n\t{self.career_stats}
Number of events:\n\t{len(self.rolling_stats)}"""

	def add_current_stats_to_rolling(self):
		"""
		Adds the snapshot of the Player's current career stats 
		to their rolling stats.
		"""

		self.rolling_stats.append(self.career_stats.copy())

	def __delete_rolling_stats(self):
		self.rolling_stats = []


class Hitter(Player):
	def __init__(self, player_id):
		super().__init__(player_id)
		self.career_stats = {
			'bb': 0,
			'k': 0,
			'h': 0,
			'sac': 0,
			'tb': 0,
			'ab': 0
		}
		self.add_current_stats_to_rolling()

	def __repr__(self):
		return f'Hitter({self.player_id})'

	def __at_bats(self, record=None):
		"""
		Return at bats from record (default: self.career_stats).

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		if record is None:
			record = self.career_stats
		return record.get('ab', 0)

	def __get_strikeout_rate(self, record=None):
		"""
		Return strikeouts / at_bat from record (default: self.career_stats).

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		if record is None:
			record = self.career_stats
		ab = self.__at_bats(record)
		if ab == 0:
			return -1.0
		return record.get('k', 0) / ab

	def __get_obp(self, record=None):
		"""
		Return on base percentage from record (default: self.career_stats).

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		if record is None:
			record = self.career_stats
		numerator = record.get('h', 0) + \
					record.get('bb', 0)
		denominator = self.__at_bats(record) + \
					  record.get('bb', 0) + \
					  record.get('sac', 0)
		if denominator == 0:
			return -1.0
		return numerator / denominator

	def __get_slg(self, record=None):
		"""
		Return total_bases / at_bat from record (default: self.career_stats).

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		if record is None:
			record = self.career_stats
		ab = self.__at_bats(record)
		if ab == 0:
			return -1.0
		return record.get('tb', 0) / ab

	def __get_ops(self, record=None):
		"""
		Return obp + slg from record (default: self.career_stats).

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		if record is None:
			record = self.career_stats
		obp = self.__get_obp(record)
		slg = self.__get_slg(record)
		if (obp == -1.0) & (slg == -1.0):
			return -1.0
		elif obp == -1.0:
			obp = 0
		elif slg == -1.0:
			slg = 0
		return obp + slg

	def get_rolling_stats_df(self):
		"""
		Return a dataframe from the Hitter's rolling_stats.
		Final columns are [`bb`, `k,` `h,` `sac`, `tb`, `ab`, `k%`, `ops`]
		and an index of [`pa`] (plate appearances).
		"""

		df_rolling = pd.DataFrame(self.rolling_stats)
		df_engineered = pd.DataFrame([self.stats_heading_in(record) 
									  for record in self.rolling_stats],
									  columns=['ab', 'k%', 'ops'])
		df = pd.merge(df_rolling, df_engineered, on='ab')
		df.index.names = ['pa']
		return df

	def stats_heading_in(self, record=None):
		"""
		Returns career stats as a snapshot to apply to the dataset.
		(`ab`, `k%`, `ops`)

		If a `record` (dict) is passed, it will be used.
		Otherwise, the current `career_stats` will be used.
		"""

		return (self.__at_bats(record), 
				self.__get_strikeout_rate(record), 
				self.__get_ops(record))

	def process_event_outcome(self, outcome, total_bases):
		"""
		Updates the Player's career_stats & 
		appends the Player's rolling_stats.
		"""

		# Update career_stats.
		at_bat = False # Some events count as at-bats and others do not.
		if outcome == 'BB':
			self.career_stats['bb'] = self.career_stats.get('bb', 0) + 1
		elif outcome == 'K':
			at_bat = True
			self.career_stats['k'] = self.career_stats.get('k', 0) + 1
		elif outcome in ('S', 'D', 'T', 'HR'):
			at_bat = True
			self.career_stats['h'] = self.career_stats.get('h', 0) + 1
			self.career_stats['tb'] = self.career_stats.get('tb', 0) + \
									  total_bases
		elif outcome == 'SAC':
			self.career_stats['sac'] = self.career_stats.get('sac', 0) + 1
		elif outcome == 'O':
			at_bat = True

		if at_bat:
			self.career_stats['ab'] = self.career_stats.get('ab', 0) + 1

		# Append their career stats as a snapshot to their rolling_stats.
		self.add_current_stats_to_rolling()


class Pitcher(Player):
	def __init__(self, player_id):
		super().__init__(player_id)
		self.career_stats = {
			'o': 0,
			'bb': 0,
			'h': 0,
			'k': 0
		}
		self.add_current_stats_to_rolling()

	def __repr__(self):
		return f'Pitcher({self.player_id})'

	def __get_ip(self):
		"""Return (outs / 3) from hitter_dct."""

		return self.career_stats.get('o', 0) / 3

	def __get_whip(self):
		"""Return (bb + h) / ip from hitter_dct."""

		ip = self.__get_ip()
		if ip == 0:
			return -1.0
		return (self.career_stats.get('bb', 0) + \
				self.career_stats.get('h', 0)) / ip

	def __get_k_bb_ratio(self):
		"""Return k / bb from hitter_dct."""

		bb = self.career_stats.get('bb', 0)
		if bb == 0:
			return -1.0
		return self.career_stats.get('k', 0) / bb

	def stats_heading_in(self):
		"""
		Returns career stats as a snapshot to apply to the dataset.
		(`ip`, `whip`, `k_bb_ratio`)
		"""

		return (self.__get_ip(), 
				self.__get_whip(), 
				self.__get_k_bb_ratio())

	def process_event_outcome(self, outcome, total_bases=None):
		"""
		Appends the Player's rolling_stats and 
		updates the Player's career_stats.
		"""
		
		# Update career_stats.
		is_out = False
		if outcome == 'BB':
			self.career_stats['bb'] = self.career_stats.get('bb', 0) + 1
		elif outcome in ('S', 'D', 'T', 'HR'):
			self.career_stats['h'] = self.career_stats.get('h', 0) + 1
		elif outcome == 'K':
			self.career_stats['k'] = self.career_stats.get('k', 0) + 1
			is_out = True
		elif outcome in ('SAC', 'O'):
			is_out = True
	        
		if is_out:
			self.career_stats['o'] = self.career_stats.get('o', 0) + 1
	    
		# Append their career stats as a snapshot to their rolling_stats.
		self.add_current_stats_to_rolling()
		