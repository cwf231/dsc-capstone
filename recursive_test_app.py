import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from baseball_support import shuffle_lst
import numpy as np
import time
import json


INIT_LAYOUT = html.Div([
	dbc.Button('Run', id='submit'),
	dbc.Button('Clear', id='clear'),
	dcc.Interval(id='interval', disabled=True),
	html.Div(id='interval-state'),

	html.H3('All Hitters'),
	html.Div(id='all-hitters'),

	html.H3('Locked In'),
	html.Div(id='locked-in'),

	html.H3('Shuffled Lineups'),
	html.Div(id='shuffled-lineups'),

	html.H3('Current Try'),
	html.Div(id='current-try'),

	html.H3('Lineup Results'),
	html.Div(id='lineup-results'),

	html.H3('Final Output'),
	html.Div(id='final-output')
])

app = dash.Dash(__name__)
app.layout = html.Div(id='main-div', children=INIT_LAYOUT)


@app.callback(
	Output('all-hitters', 'children'),
	Input('submit', 'n_clicks')
	)
def populate_hitters(n_clicks):
	if not n_clicks:
		raise PreventUpdate
	return json.dumps(['h1','h2','h3','h4','h5','h6','h7','h8','h9'])


@app.callback(
	Output('shuffled-lineups', 'children'),
	Input('interval', 'n_intervals'),
	State('all-hitters', 'children'),
	State('locked-in', 'children')
	)
def shuffle_lineups(n_intervals, all_hitters, locked_in):
	if not all_hitters:
		raise PreventUpdate
	all_hitters = json.loads(all_hitters)
	locked_in = [] if locked_in is None else json.loads(locked_in)

	order = locked_in + [h for h in all_hitters if h not in locked_in]
	start_idx = len(locked_in)

	shuffled = shuffle_lst(
		order, 
		start_idx=start_idx, 
		masked_elements=len(all_hitters)-1-start_idx
		)
	return json.dumps(shuffled)


@app.callback(
	Output('current-try', 'children'),
	Input('interval', 'n_intervals'),
	State('shuffled-lineups', 'children'),
	State('lineup-results', 'children'),
	State('current-try', 'children'),
	State('locked-in', 'children')
	)
def get_current_try(n_intervals, 
					shuffled_lineups, 
					lineup_results, 
					current_try, 
					locked_in):
	if shuffled_lineups is None:
		raise PreventUpdate
	shuffled_lineups = json.loads(shuffled_lineups)
	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	locked_in = [] if not locked_in else json.loads(locked_in)
	current_try = [] if not current_try else json.loads(current_try)

	if len(lineup_results) > len(shuffled_lineups):
		lineup_results = []
	if len(lineup_results) == len(shuffled_lineups):
		raise PreventUpdate

	next_try = shuffled_lineups[len(lineup_results)]

	if current_try == next_try:
		raise PreventUpdate
	return json.dumps(next_try)


@app.callback(
	Output('lineup-results', 'children'),
	Input('current-try', 'children'),
	State('lineup-results', 'children'),
	State('locked-in', 'children'),
	State('shuffled-lineups', 'children')
	# State(GAME_DATA)
	)
def simulate(current_try, lineup_results, locked_in, shuffled_lineups):
	if not current_try:
		raise PreventUpdate
	current_try = json.loads(current_try)

	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	locked_in = [] if not locked_in else json.loads(locked_in)
	lineups = [] if not shuffled_lineups else json.loads(shuffled_lineups)

	active_hitters = [x for x in current_try if x]
	h = active_hitters[-1] # get last hitter in lst

	# Clear if locked in has been updated.
	if len(lineup_results) > len(lineups):
		lineup_results = []

	# time.sleep(1)
	lineup_results.append(
		(h, np.random.rand(), np.random.rand())
		)
	return json.dumps(lineup_results)


@app.callback(
	Output('locked-in', 'children'),
	Input('lineup-results', 'children'),
	State('locked-in', 'children'),
	State('all-hitters', 'children')
	)
def set_locked_in(lineup_results, locked_in, all_hitters):
	if not lineup_results:
		raise PreventUpdate
	lineup_results = [] if not lineup_results else json.loads(lineup_results)
	locked_in = [] if not locked_in else json.loads(locked_in)
	all_hitters = [] if not all_hitters else json.loads(all_hitters)

	if len(lineup_results + locked_in) != len(all_hitters):
		raise PreventUpdate

	# Find player with highest expected_runs_scored.
	top_hitter = sorted(
		lineup_results, 
		key=lambda x: (x[1], x[2]), 
		reverse=True
		)[0][0]
	locked_in.append(top_hitter)
	return json.dumps(locked_in)


@app.callback(
	Output('final-output', 'children'),
	Input('shuffled-lineups', 'children'),
	State('all-hitters', 'children')
	)
def set_final_result(shuffled_lineups, all_hitters):
	lineups = [] if not shuffled_lineups else json.loads(shuffled_lineups)
	if not lineups:
		raise PreventUpdate
	all_hitters = [] if not all_hitters else json.loads(all_hitters)
	active_hitters = [x for x in lineups[0] if x]
	if len(active_hitters) == len(all_hitters):
		return json.dumps(lineups[0])
	else:
		raise PreventUpdate


@app.callback(
	Output('interval', 'disabled'),
	Output('interval-state', 'children'),
	Input('final-output', 'children'),
	Input('all-hitters', 'children')
	)
def reset_interval(final_output, all_hitters):
	if all_hitters and not final_output:
		# print('Turning ON interval.')
		return False, 'Interval timer ON.'
	else:
		# print('Turning OFF interval.')
		return True, 'Interval timer OFF.'


@app.callback(
	Output('main-div', 'children'),
	Input('clear', 'n_clicks')
	)
def reset_page(n_clicks):
	if not n_clicks:
		raise PreventUpdate
	return INIT_LAYOUT


if __name__ == '__main__':
	app.run_server(debug=True)
