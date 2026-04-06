from stripsProblem import STRIPS_domain, Strips, Planning_problem
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP
import time

boolean = {False, True}
delivery_domain = STRIPS_domain(
    {'RLoc':{'cs', 'off', 'lab', 'mr'}, 'RHC':boolean, 'SWC':boolean,
     'MW':boolean, 'RHM':boolean},           #feature:values dictionary
    { Strips('mc_cs', {'RLoc':'cs'}, {'RLoc':'off'}),   
     Strips('mc_off', {'RLoc':'off'}, {'RLoc':'lab'}),
     Strips('mc_lab', {'RLoc':'lab'}, {'RLoc':'mr'}),
     Strips('mc_mr', {'RLoc':'mr'}, {'RLoc':'cs'}),
     Strips('mcc_cs', {'RLoc':'cs'}, {'RLoc':'mr'}),   
     Strips('mcc_off', {'RLoc':'off'}, {'RLoc':'cs'}),
     Strips('mcc_lab', {'RLoc':'lab'}, {'RLoc':'off'}),
     Strips('mcc_mr', {'RLoc':'mr'}, {'RLoc':'lab'}),
     Strips('puc', {'RLoc':'cs', 'RHC':False}, {'RHC':True}),  
     Strips('dc', {'RLoc':'off', 'RHC':True}, {'RHC':False, 'SWC':False}),
     Strips('pum', {'RLoc':'mr','MW':True}, {'RHM':True,'MW':False}),
     Strips('dm', {'RLoc':'off', 'RHM':True}, {'RHM':False})
   } )

domain = STRIPS_domain(
    {'PLoc':{'home', 'work', 'car', 'park', 'shop', 'forest', 'lake'},
     'money':boolean, 'cookie':boolean, 'flour':boolean, 'relaxed':boolean, 'flower':boolean, 'seeds':boolean,
     'fish':boolean, 'axe':boolean, 'sugar':boolean, 'happy':boolean, 'wood':boolean, 'cookie_eaten':boolean, 'fish_eaten':boolean,
     'fed_birds':boolean, 'cooked_fish':boolean, 'bait':boolean, 'smoked_fish':boolean, 'smoked_fish_eaten':boolean},
    {
        Strips('home_to_car', {'PLoc':'home'}, {'PLoc':'car'}),
        Strips('car_to_home', {'PLoc':'car'}, {'PLoc':'home'}),
        Strips('work_to_car', {'PLoc':'work'}, {'PLoc':'car'}),
        Strips('car_to_work', {'PLoc':'car'}, {'PLoc':'work'}),
        Strips('park_to_car', {'PLoc':'park'}, {'PLoc':'car'}),
        Strips('car_to_park', {'PLoc':'car'}, {'PLoc':'park'}),
        Strips('shop_to_car', {'PLoc':'shop'}, {'PLoc':'car'}),
        Strips('car_to_shop', {'PLoc':'car'}, {'PLoc':'shop'}),
        Strips('forest_to_car', {'PLoc':'forest'}, {'PLoc':'car'}),
        Strips('car_to_forest', {'PLoc':'car'}, {'PLoc':'forest'}),
        Strips('lake_to_car', {'PLoc':'lake'}, {'PLoc':'car'}),
        Strips('car_to_lake', {'PLoc':'car'}, {'PLoc':'lake'}),
        Strips('fishing', {'PLoc':'lake', 'happy':True, 'bait': True}, {'fish':True, 'happy':False, 'bait':False}),
        Strips('working', {'PLoc':'work'}, {'money':True}),
        Strips('talk', {'PLoc':'work'}, {'happy':True}),
        Strips('buy_flour', {'PLoc':'shop', 'money':True}, {'money':False, 'flour':True}),
        Strips('buy_flower', {'PLoc':'shop', 'money':True}, {'money':False, 'flower':True}),
        Strips('buy_seeds', {'PLoc':'shop', 'money':True}, {'money':False, 'seeds':True}),
        Strips('buy_sugar', {'PLoc':'shop', 'money':True}, {'money':False, 'sugar':True}),
        Strips('buy_axe', {'PLoc':'shop', 'money':True}, {'money':False, 'axe':True}),
        Strips('buy_bait', {'PLoc':'shop', 'money':True}, {'money':False, 'bait':True}),
        Strips('chop_tree', {'PLoc':'forest', 'axe':True}, {'axe':False, 'wood':True}),
        Strips('relaxing', {'PLoc':'park'}, {'relaxed':True}),
        Strips('feed_birds', {'PLoc':'park', }, {'relaxed':True, 'seeds':False, 'fed_birds':True}),
        Strips('cook_fish', {'PLoc':'home', 'relaxed':True, 'fish':True}, {'relaxed':False, 'fish':False, 'cooked_fish':True}),
        Strips('bake_cookie', {'PLoc':'home', 'flour':True, 'sugar':True, 'relaxed':True},
               {'cookie':True, 'flour':False, 'sugar':False, 'relaxed':False}),
        Strips('eat_cookie', {'PLoc':'home', 'cookie':True}, {'happy':True, 'cookie':False, "cookie_eaten":True}),
        Strips('eat_fish', {'PLoc':'home', 'cooked_fish':True}, {'cooked_fish':False, "fish_eaten":True}),
        Strips('smoke_fish', {'PLoc':'home', 'fish':True, 'wood':True}, {'wood':False, 'fish':True, 'smoked_fish':True}),
        Strips('eat_smoked_fish', {'PLoc':'home', 'smoked_fish':True}, {'smoked_fish':False, "smoked_fish_eaten":True}),

        
    }
)

start_state = {'PLoc':'home', 'money':False, 'cookie':False, 'flour':False, 'sugar':False, 'relaxed':False, 'happy':False,
                             'flower':False, 'seeds':False, 'fish':False, 'cooked_fish':False, 'axe':False, 'fish_eaten':False, 'cookie_eaten':False,
                             'wood':False, 'fed_birds':False, 'bait':False, 'smoked_fish':False, 'smoked_fish_eaten':False}

problem1 = Planning_problem(domain,
                            start_state,
                            {'cookie':True})

problem2 = Planning_problem(domain,
                            start_state,
                            {'happy':True})

def h1(state, goal):
    h = 0
    for var, target_val in goal.items():
        if state.get(var) != target_val:
            h += 1
            # Jeśli celem jest zjedzenie ciastka, a go nie mamy:
            if var == 'cookie_eaten' and not state.get('cookie'):
                h += 1 # Musimy upiec
                if not state.get('flour') or not state.get('sugar'):
                    h += 1 # Musimy kupić składniki
            # Jeśli celem jest mąka/cukier, a nie mamy kasy:
            if (var == 'flour' or var == 'sugar') and not state.get('money'):
                h += 1
    return h

def h2(state, goal):
    h = 0
    for var, target_val in goal.items():
        if state.get(var) != target_val:
            h += 1
            # Droga do zjedzenia ryby
            if var == 'fish_eaten' and not state.get('cooked_fish'):
                h += 1
                if not state.get('fish'):
                    h += 1
            # Droga do złowienia ryby (potrzeba happy i bait)
            if var == 'fish':
                if not state.get('happy'): h += 1
                if not state.get('bait'): h += 1
    return h

def h3(state, goal):
    h = 0
    for var, target_val in goal.items():
        if state.get(var) != target_val:
            h += 1
            # Główny cel: zjedzenie wędzonej ryby
            if var == 'smoked_fish_eaten' and not state.get('smoked_fish'):
                h += 1
                # Brakujące składniki do wędzenia:
                if not state.get('fish'): h += 1
                if not state.get('wood'): h += 1
            
            # Droga do drewna
            if var == 'wood' and not state.get('axe'):
                h += 1
                if not state.get('money'): h += 1
                
            # Droga do ryby (pieniądze na przynętę)
            if var == 'fish' and not state.get('money') and not state.get('bait'):
                h += 1
    return h

def h4(state, goal):
    if 'fed_birds' in goal:
        if state.get('fed_birds'): return 0
        if state.get('seeds'): return 1
        return 2
    if 'seeds' in goal:
        return 0 if state.get('seeds') else 1
    return 0

def h5(state, goal):
    if 'flour' in goal:
        if state.get('flour'): return 0
        if state.get('money'): return 1
        return 2
    if 'money' in goal:
        return 0 if state.get('money') else 1
    if 'PLoc' in goal:
        return 0 if state.get('PLoc') == goal['PLoc'] else 1
    return 0

def h6(state, goal):
    if 'flower' in goal:
        if state.get('flower'): return 0
        if state.get('money'): return 1
        return 2
    if 'money' in goal:
        return 0 if state.get('money') else 1
    if 'PLoc' in goal:
        return 0 if state.get('PLoc') == goal['PLoc'] else 1
    return 0


# PODCELE (podejście sekwencyjne)
problem1 = [
    {'money': True},    
    {'flour': True, 'sugar': True},
    {'cookie_eaten': True},
    {'flour': True, 'sugar': True},
    {'cookie_eaten': True, 'cookie': True}
]

problem2 = [
    {'happy': True},
    {'fish': True},
    {'fish_eaten': True},
    {'fish': True},
    {'fish_eaten': True, 'fish': True}
]
problem3 = [
    {'money': True},
    {'axe': True},
    {'wood': True},
    {'fish': True},
    {'smoked_fish_eaten': True},
]
problem4 = [
    {'money': True},
    {'seeds': True},
    {'fed_birds': True}
]
problem5 = [
    {'PLoc': 'work'},
    {'money': True},
    {'flour': True}
]
problem6 = [
    {'PLoc': 'work'},
    {'money': True},
    {'flower': True}
]

def solve_subproblems(subproblems, heur_function) -> tuple[float, float, str]:
    global start_state, domain
    current_state = start_state
    current_state_h = start_state
    all_actions = []
    time_norm = 0.0
    time_heur = 0.0
    for goal in subproblems:
        sub_problem = Planning_problem(domain, current_state, goal)
        tic = time.time()
        searcher = SearcherMPP(Forward_STRIPS(sub_problem))
        searcher.max_display_level = 0
        result = searcher.search()
        
        toc = time.time()

        if result:
            time_norm += (toc - tic)
            current_state = result.end().assignment
            kroki_etapu = []
            
            p = result
            while p is not None and p.arc is not None:
                if p.arc.action:
                    kroki_etapu.append(p.arc.action.name)
                p = p.initial
            
            kroki_etapu.reverse()
            
            all_actions.extend(kroki_etapu)
        else:
            print(f"Błąd: Nie znaleziono rozwiązania dla celu {goal}")
            return time_norm, time_heur

        sub_problem = Planning_problem(domain, current_state_h, goal)
        tic = time.time()
        searcher = SearcherMPP(Forward_STRIPS(sub_problem, heur_function))
        searcher.max_display_level = 0
        result_h = searcher.search()
        toc = time.time()

        if result_h:
            time_heur += (toc - tic)
            current_state_h = result_h.end().assignment
        else:
            print(f"Błąd: Nie znaleziono rozwiązania (heur) dla celu {goal}")
            return time_norm, time_heur
    full_actions = ',\n'.join(all_actions)
    return time_norm, time_heur, full_actions

problems = [problem1, problem2, problem3, problem4, problem5, problem6]
heuristics = [h1, h2, h3, h4, h5, h6]

for i, (problem, heur) in enumerate(zip(problems, heuristics)):
    print(f"\n\nProblem {i+1}")
    time_norm, time_heur, actions = solve_subproblems(problem, heur)
    print(f"Czas normalny: {time_norm:.5f}, czas heurystyki: {time_heur:.5f}")
    print(f"Akcje:\n{actions}")

