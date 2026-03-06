from easyAI import AI_Player, Human_Player, Negamax
from easyAI.games.Hexapawn import Hexapawn
from hexapawn_prob import HexapawnProb

scoring = lambda game: -100 if game.lose() else 0
ai = Negamax(10, scoring)

player1 = AI_Player(ai)
player2 = AI_Player(ai)

num_games = 10
player1_wins = 0
player2_wins = 0


for i in range(num_games):
    game = Hexapawn([player1, player2])
    game.current_player = 1 + i%2
    game.play(nmoves=1, verbose=False)

    if game.opponent_index == 1:
        player1_wins += 1
    elif game.opponent_index == 2:
        player2_wins += 1
    else:
        raise ValueError("Invalid opponent index")

print("Default Hexspawn, number of games: ", num_games)
print("Player 1 wins: ", player1_wins)
print("Player 2 wins: ", player2_wins)
print("Draws: ", num_games - player1_wins - player2_wins)

for i in range(num_games):
    game = HexapawnProb([player1, player2])
    game.current_player = 1 + i%2
    game.play(nmoves=1, verbose=False)

    if game.opponent_index == 1:
        player1_wins += 1
    elif game.opponent_index == 2:
        player2_wins += 1
    else:
        raise ValueError("Invalid opponent index")

print("HexapawnProb, number of games: ", num_games)
print("Player 1 wins: ", player1_wins)
print("Player 2 wins: ", player2_wins)
print("Draws: ", num_games - player1_wins - player2_wins)