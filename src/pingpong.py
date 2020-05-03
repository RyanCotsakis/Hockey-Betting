from scipy.misc import comb
import numpy as np
import matplotlib.pyplot as plt
import os

# Return the probability that player 1 wins the set
def who_wins(score1, score2, playto, p=0.5):
	if score1 == playto:
		return 1.0
	prob = 0
	k = playto - 1 - score1
	largest = k + (playto - 1 - score2)

	for n in range(k, largest + 1):
		prob += comb(n,k) * p**(k+1) * (1-p)**(n-k)

	return prob


# Return the probability that player 1 wins the match given the current score, and the number of sets won
def win_match(score1, score2, sets1, sets2, p=0.5, match_playto=3, set_playto=11):
	if sets1 == match_playto:
		return 1.0
	prob = 0
	prob += who_wins(sets1 + 1, sets2, match_playto, who_wins(0, 0, set_playto, p)) * who_wins(score1, score2, set_playto, p)
	prob += who_wins(sets1, sets2 + 1, match_playto, who_wins(0, 0, set_playto, p)) * who_wins(score2, score1, set_playto, 1-p)
	return prob


# Prints the Vegas odds of player 1 winning, and player 2 winning
def print_odds(prob, maxi=100, bank=50):
	bet_size = lambda odds: bank * 0.1 / (2 * odds * (odds-1))
	winner1 = prob
	winner2 = 1 - winner1

	if winner1:
		winner1 = max(min(1.0/winner1, maxi), 1.01)
	else:
		winner1 = maxi

	if winner2:
		winner2 = max(min(1.0/winner2, maxi), 1.01)
	else:
		winner2 = maxi


	print("{:.2f} : {:.2f}".format(winner1, winner2))
	print("		${:.2f} : ${:.2f}".format(bet_size(winner1), bet_size(winner2)))
	print("		Bet the above amount times 10*(line-odds)")



def newgame(odds=None):
	os.system('clear')
	score1 = 0
	score2 = 0
	sets1 = 0
	sets2 = 0
	if odds is not None:
		p = (1./odds[0] - 1./odds[1] + 3.8) / 7.6
	else:
		p = 0.5

	while True:
		print("Line start: {}".format(odds))
		print("Match Score: {}:{}".format(sets1,sets2))
		print("Set Score: {}:{}".format(score1,score2))
		print()

		print("Odds for the match:")
		print_odds(win_match(score1, score2, sets1, sets2, p=p))
		print()

		print("Odds for this set:")
		print_odds(who_wins(score1, score2, 11, p=p))
		print()

		query = input('>> ')
		os.system('clear')

		for c in query:
			if c == 'h':
				sets1 += 1
				score1 = 0
				score2 = 0
			elif c == 'l':
				sets2 += 1
				score1 = 0
				score2 = 0
			elif c == 'j':
				score1 += 1
			elif c == 'k':
				score2 += 1
			elif c == 'c':
				score1 = 0
				score2 = 0
				sets1 = 0
				sets2 = 0
			elif c == 'q':
				return
			elif c == ',':
				newgame([float(o) for o in query.split(',')])
				return


newgame()
