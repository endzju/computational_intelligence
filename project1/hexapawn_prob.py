from dataclasses import dataclass
import random
from typing import override

from easyAI.games.Hexapawn import Hexapawn

@dataclass
class Pawn:
    cur_pos: tuple[int, int]
    start_pos: tuple[int, int]


class HexapawnProb(Hexapawn):
    @override
    def __init__(self, players, size=(4, 4), respawn_prob=0.1):
        self.size = M, N = size
        p: list[list[Pawn]] = [[Pawn((i, j), (i, j)) for j in range(N)] for i in [0, M - 1]]

        for i, d, goal, pawns in [(0, 1, M - 1, p[0]), (1, -1, 0, p[1])]:
            players[i].direction = d
            players[i].goal_line = goal
            players[i].pawns = pawns

        self.players = players
        self.current_player = 1
        self.captured_pawns = []
        self.respawn_prob = respawn_prob
        self.move_count = 0

    @override
    def possible_moves(self):
        moves = []
        opponent_pawns = self.opponent.pawns
        d = self.player.direction
        for p in self.player.pawns:
            i, j = p.cur_pos

            if not any((i + d, j) == p.cur_pos for p in opponent_pawns):
                moves.append(((i, j), (i + d, j)))
            if any((i + d, j + 1) == p.cur_pos for p in opponent_pawns):
                moves.append(((i, j), (i + d, j + 1)))
            if any((i + d, j - 1) == p.cur_pos for p in opponent_pawns):
                moves.append(((i, j), (i + d, j - 1)))

        return list(map(self._to_string, [(i, j) for i, j in moves]))

    @override
    def make_move(self, move, prob=True):
        move = list(map(self._to_tuple, move.split(" ")))
        ind = next((i for i, p in enumerate(self.player.pawns) if p.cur_pos == move[0]), -1)
        self.player.pawns[ind].cur_pos = move[1]

        if any(move[1] == p.cur_pos for p in self.opponent.pawns):
            p = self.opponent.pawns.pop(next((i for i, p in enumerate(self.opponent.pawns) if p.cur_pos == move[1]), -1))
            self.captured_pawns.append(p)
        
        if not self.is_over():
            curr_player_start = 0 if self.current_player == 1 else self.size[0] - 1
            curr_player_captured = [pawn for pawn in self.captured_pawns if pawn.start_pos[0] == curr_player_start]

            if prob and curr_player_captured and random.random() <= self.respawn_prob:
                p = random.choice(curr_player_captured)
                p.cur_pos = p.start_pos
                self.player.pawns.append(p)
        
        self.move_count += 1
    
    def get_num_prob_states(self):
        curr_player_start = 0 if self.current_player == 1 else self.size[0] - 1
        curr_player_captured = [pawn for pawn in self.captured_pawns if pawn.start_pos[0] == curr_player_start]
        
        return len(curr_player_captured)
    
    def apply_prob_state(self, i):
        curr_player_start = 0 if self.current_player == 1 else self.size[0] - 1
        curr_player_captured = [pawn for pawn in self.captured_pawns if pawn.start_pos[0] == curr_player_start]

        if curr_player_captured:
            p = curr_player_captured[i]
            p.cur_pos = p.start_pos
            self.player.pawns.append(p)


    @override
    def lose(self):
        return any([p.cur_pos[0] == self.opponent.goal_line for p in self.opponent.pawns]) or (
            self.possible_moves() == []
        )

    @override
    def is_over(self):
        return self.lose()

    @override
    def show(self):
        f = (
            lambda x: "1"
            if any(x == p.cur_pos for p in self.players[0].pawns)
            else ("2" if any(x == p.cur_pos for p in self.players[1].pawns) else ".")
        )
        print(
            "\n".join(
                [
                    " ".join([f((i, j)) for j in range(self.size[1])])
                    for i in range(self.size[0])
                ]
            )
        )
    
    def _to_string(self, move):
        return " ".join(
            ["ABCDEFGHIJ"[move[i][0]] + str(move[i][1] + 1) for i in (0, 1)]
        )
    
    def _to_tuple(self, s):
        return ("ABCDEFGHIJ".index(s[0]), int(s[1:]) - 1)


if __name__ == "__main__":
    from easyAI import AI_Player, Human_Player, Negamax

    scoring = lambda game: -100 if game.lose() else 0
    ai = Negamax(10, scoring)
    game = HexapawnProb([AI_Player(ai), AI_Player(ai)])
    game.play()
    print("player %d wins after %d turns " % (game.opponent_index, game.nmove))
