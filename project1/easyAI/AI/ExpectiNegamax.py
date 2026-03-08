"""
The standard AI algorithm of easyAI is Negamax with alpha-beta pruning
and (optionnally), transposition tables.
"""

import pickle

from hexapawn_prob import HexapawnProb

LOWERBOUND, EXACT, UPPERBOUND = -1, 0, 1
inf = float("infinity")


def negamax(game, depth, origDepth, scoring, alpha=-inf, beta=+inf, tt=None, use_ab=False):
    alphaOrig = alpha

    lookup = None if (tt is None) else tt.lookup(game)

    if lookup is not None:
        if lookup["depth"] >= depth:
            flag, value = lookup["flag"], lookup["value"]
            if flag == EXACT:
                if depth == origDepth:
                    game.ai_move = lookup["move"]
                return value
            elif flag == LOWERBOUND:
                alpha = max(alpha, value)
            elif flag == UPPERBOUND:
                beta = min(beta, value)

            if use_ab:
                if alpha >= beta:
                    if depth == origDepth:
                        game.ai_move = lookup["move"]
                    return value

    if (depth == 0) or game.is_over():
        return scoring(game) * (1 + 0.001 * depth)

    if lookup is not None:
        possible_moves = game.possible_moves()
        possible_moves.remove(lookup["move"])
        possible_moves = [lookup["move"]] + possible_moves

    else:
        possible_moves = game.possible_moves()

    state = game
    best_move = possible_moves[0]
    if depth == origDepth:
        state.ai_move = possible_moves[0]

    bestValue = -inf
    unmake_move = hasattr(state, "unmake_move")

    for move in possible_moves:

        if not unmake_move:
            game = state.copy()

        if not isinstance(game, HexapawnProb) or (n_options := game.get_num_prob_states()) == 0:
            game.make_move(move)
            game.switch_player()

            move_alpha = -negamax(game, depth - 1, origDepth, scoring, -beta, -alpha, tt, use_ab)
        else:
            U = 0
            L = -100.0

            game.make_move(move, prob=False)
            game.switch_player()
            
            bound_alpha = (alpha - 0.1 * U) / 0.9 if alpha != -inf else -inf
            bound_beta = (beta - 0.1 * L) / 0.9 if beta != inf else inf

            v_90 = -negamax(game, depth - 1, origDepth, scoring, -bound_beta, -bound_alpha, tt, use_ab)
            expected_value = 0.9 * v_90
            
            prob_step = 0.1 / n_options
            
            for i in range(n_options):
                remaining_prob = 0.1 - ((i + 1) * prob_step)
                
                current_bound_alpha = (alpha - expected_value - (remaining_prob * U)) / prob_step if alpha != -inf else -inf
                current_bound_beta = (beta - expected_value - (remaining_prob * L)) / prob_step if beta != inf else inf

                game_curr = game.copy()
                game_curr.switch_player()
                game_curr.apply_prob_state(i)
                game_curr.switch_player()

                v_i = -negamax(game_curr, depth - 1, origDepth, scoring, -current_bound_beta, -current_bound_alpha, tt, use_ab)
                expected_value += prob_step * v_i
                
            move_alpha = expected_value

        if unmake_move:
            game.switch_player()
            game.unmake_move(move)

        if bestValue < move_alpha:
            bestValue = move_alpha
            best_move = move

        if alpha < move_alpha:
            alpha = move_alpha
            if depth == origDepth:
                state.ai_move = move
            if use_ab:
                if alpha >= beta:
                    break

    if tt is not None:
        assert best_move in possible_moves
        tt.store(
            game=state,
            depth=depth,
            value=bestValue,
            move=best_move,
            flag=UPPERBOUND
            if (bestValue <= alphaOrig)
            else (LOWERBOUND if (bestValue >= beta) else EXACT),
        )

    return bestValue


class ExpectiNegamax:
    def __init__(self, depth, scoring=None, win_score=+inf, tt=None, use_ab=False):
        self.scoring = scoring
        self.depth = depth
        self.tt = tt
        self.win_score = win_score
        self.use_ab = use_ab

    def __call__(self, game):
        scoring = (
            self.scoring if self.scoring else (lambda g: g.scoring())
        ) 

        self.alpha = negamax(
            game,
            self.depth,
            self.depth,
            scoring,
            -self.win_score,
            +self.win_score,
            self.tt,
            self.use_ab,
        )
        return game.ai_move