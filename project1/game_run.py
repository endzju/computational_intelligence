from easyAI import AI_Player, Human_Player, Negamax
from easyAI.AI.ExpectiNegamax import ExpectiNegamax
from easyAI.games.Hexapawn import Hexapawn
from hexapawn_prob import HexapawnProb
import time


def play_game(game_class, ai, num_games, nmoves = 50) -> tuple[int, int]:
    
    player1_wins = 0
    player2_wins = 0
    move_count = 0
    start = time.time()
    for i in range(num_games):
        game = game_class([AI_Player(ai), AI_Player(ai)], size=(5,5))
        game.current_player = 1 + i%2
        game.play(nmoves=nmoves, verbose=False)

        if game.opponent_index == 1:
            player1_wins += 1
        elif game.opponent_index == 2:
            player2_wins += 1
        else:
            raise ValueError("Invalid opponent index")
        
        move_count += game.move_count
    end = time.time()
    print(f"Class: {game_class}")
    print("Number of games: ", num_games)
    print(f"Time: {end - start:.4f} s")
    print(f"Time per move: {(end - start)/move_count:.4f} s")
    print("Player 1 wins: ", player1_wins)
    print("Player 2 wins: ", player2_wins)


scoring = lambda game: -100 if game.lose() else 0

num_games = 10
depth = 3
ai = ExpectiNegamax(depth, scoring)
play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("without alpha-beta pruning")
print()
play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("without alpha-beta pruning")
print()

depth = 5
ai = ExpectiNegamax(depth, scoring)
play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("without alpha-beta pruning")
print()
play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("without alpha-beta pruning")
print()

depth = 3
ai = ExpectiNegamax(depth, scoring, use_ab=True)
play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("with alpha-beta pruning")
print()
play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("with alpha-beta pruning")
print()

depth = 5
ai = ExpectiNegamax(depth, scoring, use_ab=True)
play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("with alpha-beta pruning")
print()
play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
print(f"Depth: {depth}")
print("with alpha-beta pruning")
print()

# depth = 3
# ai = Negamax(depth, scoring)
# play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("without alpha-beta pruning")
# print()
# play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("without alpha-beta pruning")
# print()

# depth = 5
# ai = Negamax(depth, scoring)
# play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("without alpha-beta pruning")
# print()
# play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("without alpha-beta pruning")
# print()

# depth = 3
# ai = Negamax(depth, scoring, use_ab=True)
# play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("with alpha-beta pruning")
# print()
# play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("with alpha-beta pruning")
# print()

# depth = 5
# ai = Negamax(depth, scoring, use_ab=True)
# play_game(game_class=Hexapawn, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("with alpha-beta pruning")
# print()
# play_game(game_class=HexapawnProb, ai=ai, num_games=num_games)
# print(f"Depth: {depth}")
# print("with alpha-beta pruning")
# print()