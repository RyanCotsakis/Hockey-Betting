import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import csv

class Game():
	def __init__(self, arr):
		self.visitor = arr[1]
		self.visitor_goals = int(arr[2])
		self.home = arr[3]
		self.home_goals = int(arr[4])
		self.ot =  bool(arr[5])
		self.att = int(arr[6])


	def winner(self):
		if self.home_goals > self.visitor_goals:
			return self.home
		else:
			return self.visitor


	def update_team_stats(self):
		if self.winner() == self.home:
			team_stats[self.home]["HW"] += 1
			if self.ot:
				team_stats[self.visitor]["VOL"] += 1
			else:
				team_stats[self.visitor]["VL"] += 1
		else:
			team_stats[self.visitor]["VW"] += 1
			if self.ot:
				team_stats[self.home]["HOL"] += 1
			else:
				team_stats[self.home]["HL"] += 1
		team_stats[self.home]["HGF"] += self.home_goals
		team_stats[self.home]["HGA"] += self.visitor_goals
		team_stats[self.visitor]["VGA"] += self.home_goals
		team_stats[self.visitor]["VGF"] += self.visitor_goals


	def stats(self):
		return self.home, self.visitor, self.home_goals, self.visitor_goals


def read_file(fname, header=True):
	data = []
	print("Reading from " + fname)

	with open(fname) as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			if header:
				header = False
				continue
			data.append(row)
	return np.array(data)


class Skater():
	def __init__(self, arr):
		self.id = int(arr[0])
		self.team = arr[3]

		self.toi = int(arr[21])

		self.G = int(arr[6])
		self.A = int(arr[7])
		self.plus_minus = int(arr[9])
		self.pen_minutes = int(arr[10])
		self.shots = int(arr[19])
		if self.shots == 0:
			self.shot_percent = 0
		else:
			self.shot_percent = self.G / self.shots

		self.blk = int(arr[23])
		self.hit = int(arr[24])

	def stats(self):
		return [
			self.G / np.sqrt(self.toi),
			self.A / np.sqrt(self.toi),
			self.plus_minus / np.sqrt(self.toi),
			self.pen_minutes / np.sqrt(self.toi),
			self.shots / np.sqrt(self.toi),
			self.shot_percent,
			self.blk / np.sqrt(self.toi),
			self.hit / np.sqrt(self.toi)].copy()

		
class Goalie():
	def __init__(self,arr):
		self.id = int(arr[0])
		self.team = arr[3]

		self.toi = int(arr[16])

		self.GP = int(arr[4])
		self.W = int(arr[6])
		self.L = int(arr[7])
		self.OL = int(arr[8])

		self.saves = int(arr[11])
		self.shots = int(arr[10])
		if self.shots == 0:
			self.save_percentage = 0
		else:
			self.save_percentage = self.saves / self.shots

	def stats(self):
		return [
			self.W / np.sqrt(self.GP),
			self.L / np.sqrt(self.GP),
			self.OL / np.sqrt(self.GP),
			self.save_percentage].copy()


fname = '../schedule.csv'
games = read_file(fname)

fname = '../skaters.csv'
skaters = read_file(fname)

fname = '../goalies.csv'
goalies = read_file(fname)

teams = set(games[:,1])
teams = list(teams)

long_team_name = {
	'EDM': 'Edmonton Oilers',
	'PIT': 'Pittsburgh Penguins',
	'BUF': 'Buffalo Sabres',
	'NYI': 'New York Islanders',
	'ARI': 'Arizona Coyotes',
	'VAN': 'Vancouver Canucks',
	'CHI': 'Chicago Blackhawks',
	'ANA': 'Anaheim Ducks',
	'DAL': 'Dallas Stars',
	'WSH': 'Washington Capitals',
	'DET': 'Detroit Red Wings',
	'FLA': 'Florida Panthers',
	'NSH': 'Nashville Predators',
	'PHI': 'Philadelphia Flyers',
	'NYR': 'New York Rangers',
	'VEG': 'Vegas Golden Knights',
	'CBJ': 'Columbus Blue Jackets',
	'TBL': 'Tampa Bay Lightning',
	'WPG': 'Winnipeg Jets',
	'COL': 'Colorado Avalanche',
	'TOR': 'Toronto Maple Leafs',
	'OTT': 'Ottawa Senators',
	'BOS': 'Boston Bruins',
	'NJD': 'New Jersey Devils',
	'MTL': 'Montreal Canadiens',
	'LAK': 'Los Angeles Kings',
	'CAR': 'Carolina Hurricanes',
	'STL': 'St. Louis Blues',
	'CGY': 'Calgary Flames',
	'SJS': 'San Jose Sharks',
	'MIN': 'Minnesota Wild'
}
long_team_name['TOT'] = None

team_stats = {}
stats_template = {
	"HW": 0,		#Home Wins
	"VW": 0,		#Visitor Wins
	"HL": 0,		#Home Loss (before overtime)
	"VL": 0,		#Visitor Loss (before overtime)
	"HOL": 0,		#Home Overtime Loss
	"VOL": 0,		#Visitor Overtime Loss
	"HGF": 0,		#Total Goals For as Home
	"VGF": 0,		#Total Goals For as Visitor
	"HGA": 0,		#Total Goals Against as Home
	"VGA": 0		#Total Goals Against as Visitor
}
for team in teams:
	team_stats[team] = stats_template.copy()

games = [Game(g) for g in games]
defenders = [Skater(sk) for sk in skaters if sk[4] == 'D']
forwards = [Skater(sk) for sk in skaters if sk[4] != 'D']
goalies = [Goalie(g) for g in goalies]

for game in games:
	game.update_team_stats()


# takes a list of skater ids and refines player_list to output a stat (list object) corresponding to the average player
# otherwise just uses the whole team
def create_average_player(team, player_list, l=None):
	if l is not None:
		l = [p for p in player_list if p.id in l]
	else:
		l = [p for p in player_list if long_team_name[p.team] == team]

	result = np.zeros_like(l[0].stats())
	total_toi = 0
	for p in l:
		result += np.array(p.stats()) * p.toi
		total_toi += p.toi
	return list(result / total_toi)


# pmf of a poisson random variable with lambda distrubuted as a log normal with mean mu_observed and variance s2_observed
def norm_pois_pmf(x, mu_observed, s2_observed):
	N = 1001

	# parameters of a lognormal with correct mean and variance
	s2 = np.log(s2_observed / mu_observed**2 + 1)
	mu = np.log(mu_observed) - s2/2

	b = mu_observed + 4*np.sqrt(s2_observed)  # upper limit of integration

	f = lambda l: np.exp(-l - (np.log(l)-mu)**2 / (2 * s2)) * np.math.pow(l,x-1) / np.math.factorial(x) / np.sqrt(2 * np.pi * s2)
	t = np.linspace(0, b, N)
	delta = b/(N-1)

	return np.sum([f(x) for x in t[1:]]) * delta


# Returns a grid of probabilities of the game ending in certain scores (rows correspond to home team, columns to visitor)
def compute_odds(mu_home, mu_visitor, condition, MSE=1.1, grid_size=20):
	# Creates a table of probabilities of possible scores
	pmf_home = np.array([norm_pois_pmf(x,mu_home, MSE) for x in range(grid_size)])
	pmf_visitor = np.array([norm_pois_pmf(x,mu_visitor, MSE) for x in range(grid_size)])

	tie_factor = mu_home / (mu_home + mu_visitor)

	grid = np.zeros((grid_size, grid_size))
	for i in range(grid_size):
		for j in range(grid_size):
			grid[i,j] += pmf_home[i] * pmf_home[j]
			if i == j and j < grid_size-1:
				grid[i+1,j] = tie_factor * grid[i,j]
				grid[i,j+1] = (1 - tie_factor) * grid[i,j]
				grid[i,j] = 0

	probability = 0
	for i in range(grid_size):
		for j in range(grid_size):
			if condition(i,j):
				probability += grid[i,j]

	return probability


class NadayaraWatson():

	# distance metric
	def d(x1, x2):
		return np.linalg.norm(x1 - x2)


	# Input a matrix, and output a new matrix with linearly scaled columns
	# such that each column has mean 0 and var 1
	def standard_normal_columns(X):
		M = np.array(X)  # copy
		d = len(np.shape(M))
		if d == 2:
			return np.array([(col - np.mean(col)) / np.sqrt(np.var(col)) for col in M.T]).T
		if d == 1:
			return (M - np.mean(M)) / np.sqrt(np.var(M))
		raise Exception()


	# Return the loo MSE
	def loo(X, Y, h, N=None, verbose=False):
		n = len(Y)
		Y_hat = []
		Y_correct = []

		print()
		print("Starting LOO error estimate!")
		print("h: {}".format(h))

		failures = []

		if N is None:
			N = n
		for test in range(N):
			if N == n:
				i = test
			else:
				i = np.random.randint(0,n)

			if i % np.max((N//100, 1)) == 0 and verbose:
				print("{}% complete...".format((100*test)//N))

			X_loo = np.delete(X, i, 0)
			Y_loo = np.delete(Y, i)
			estimator = NadayaraWatson(X_loo, Y_loo, h)

			try:
				Y_hat.append(estimator.predict(X[i]))
				Y_correct.append(Y[i])
				failures.append(0)
			except:
				failures.append(1)

		Y_hat = np.array(Y_hat)
		Y_correct = np.array(Y_correct[failures == 0])
		MSE = np.mean((Y_correct - Y_hat)**2)
		print("Complete!")
		print("Number of tests: {}".format(N))
		print("Failed tests: {}".format(np.sum(failures)))
		print("h: {}".format(h))
		print("Variance of estimator: {}".format(np.var(Y_hat)))
		print("MSE: {}".format(MSE))
		return MSE


	def __init__(self, X, Y, h=2):
		X1 = NadayaraWatson.standard_normal_columns(X)  # copy
		
		n, m = np.shape(X)
		H = np.eye(n) - np.ones((n, n)) / n
		S = X1.T @ H @ X1 / n
		D = np.diag(X1.T @ NadayaraWatson.standard_normal_columns(Y)) / n

		self.Y = np.array(Y)
		self.X_bar = np.mean(X, axis=0)
		self.X_var = np.var(X, axis=0)
		self.R = D @ S
		self.X = X1 @ self.R
		self.N = n
		self.h = h


	# kernel function of distance
	def K(self, x1, x2):
		return np.max((0, self.h - NadayaraWatson.d(x1, x2)))

	# predict Y based on x
	def predict(self, x, verbose=False):
		x = np.array(x)
		x = ((x - self.X_bar) / np.sqrt(self.X_var)) @ self.R

		weights = np.array([self.K(x, x2) for x2 in self.X])
		nnz = np.count_nonzero(weights)
		if verbose:
			print("Including {} of the points for the N-W estimation.".format(nnz))
		if nnz:
			return np.dot(weights, self.Y) / np.sum(weights)
		else:
			raise Exception("No points included in the N-W estimation! 'h' is too small.")



# Create a datapoint x (list of features). The None args can be lists of player indices
def create_x(home, visitor, home_forwards=None, home_defenders=None, home_goalies=None, visitor_forwards=None, visitor_defenders=None, visitor_goalies=None):
	a = [val for key, val in team_stats[home].items()]
	b = [val for key, val in team_stats[visitor].items()]
	c = create_average_player(home, forwards, home_forwards)
	d = create_average_player(visitor, forwards, visitor_forwards)
	e = create_average_player(home, defenders, home_defenders)
	f = create_average_player(visitor, defenders, visitor_defenders)
	g = create_average_player(home, goalies, home_goalies)
	h = create_average_player(visitor, goalies, visitor_goalies)
	return a + b + c + d + e + f + g + h


X = []
Y_home = []
Y_visitor = []
for game in games:
	home, visitor, home_goals, visitor_goals = game.stats()
	X.append(create_x(home, visitor))
	Y_home.append(home_goals)
	Y_visitor.append(visitor_goals)

	alpha = 0.75
	if np.random.rand() < alpha:  # Add the symmetric entry with probability alpha
		home, visitor, home_goals, visitor_goals = visitor, home, visitor_goals, home_goals
		X.append(create_x(home, visitor))
		Y_home.append(home_goals)
		Y_visitor.append(visitor_goals)


home_estimator = NadayaraWatson(X, Y_home, h=2)
visitor_estimator = NadayaraWatson(X, Y_visitor, h=2)

home_team = long_team_name['TOR']
visitor_team = long_team_name['VAN']
x = create_x(home_team, visitor_team)

mu_home = home_estimator.predict(x, verbose=True)
mu_visitor = visitor_estimator.predict(x, verbose=True)
condition = lambda i, j: np.abs(i - j) == 1
odds = compute_odds(mu_home, mu_visitor, condition)
print("Simulating {} at {}.".format(visitor_team, home_team))
print("Predicted score: ({:3.2f}, {:3.2f})".format(mu_visitor, mu_home))
print("Odds of given condition: {}".format(odds))

"""
-------------------------------------------------------
e = []
h = [1.9, 2.1]
for hh in h:
	e.append(NadayaraWatson.loo(X, Y_home, hh, verbose=True))
plt.plot(h,e)
plt.show()

------- HOME --------
Starting LOO error estimate!
h: 1.9
Complete!
Fails: 0
h: 1.9
Variance: 0.09716847536481521
MSE: 1.4199531248255177

Starting LOO error estimate!
h: 2.1
Complete!
Fails: 0
h: 2.1
Variance: 0.08801052803928254
MSE: 1.405954545465339

-------- SYMMETRIC (alpha = 1) ----------
Fails: 0
h: 1.9
Variance: 0.07436608173554196
MSE: 1.1004652249629905

Fails: 0
h: 2.1
Variance: ??
MSE: 1.095 ish
-------------------------------------------------------
"""