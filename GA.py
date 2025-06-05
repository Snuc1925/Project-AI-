import pygame
import sys
from pygame import gfxdraw
import os
from collections import defaultdict
import random
import copy
# import random # random is already imported
import time



# --- GA Utility Functions (already in your provided code) ---
def extract_edge_set(lines_drawn_tuples): # Expects list of tuples
    return set(lines_drawn_tuples)

def edges_of_square(square_nodes):
    p1, p2, p3, p4 = square_nodes
    return [
        tuple(sorted((p1, p2))), tuple(sorted((p3, p4))),
        tuple(sorted((p1, p3))), tuple(sorted((p2, p4)))
    ]

def is_dangerous_move(new_edge_tuple, current_lines_drawn_tuples, squares_config):
    temp_edges_set = set(current_lines_drawn_tuples)
    temp_edges_set.add(new_edge_tuple)
    for square_nodes in squares_config:
        edges_for_this_square = edges_of_square(square_nodes)
        if new_edge_tuple in edges_for_this_square:
            count_filled_edges = sum(1 for edge in edges_for_this_square if edge in temp_edges_set)
            if count_filled_edges == 3:
                return True
    return False

def check_new_squares(new_edge_tuple, current_lines_drawn_tuples, squares_config, completed_squares_indices_set=None):
    # current_lines_drawn_tuples is the state *before* adding new_edge_tuple
    if completed_squares_indices_set is None:
        completed_squares_indices_set = set()

    temp_edges_set = set(current_lines_drawn_tuples)
    temp_edges_set.add(new_edge_tuple) # Simulate adding the new edge

    newly_completed_indices = []
    for idx, square_nodes in enumerate(squares_config):
        if idx in completed_squares_indices_set:
            continue
        edges_for_this_square = edges_of_square(square_nodes)
        if new_edge_tuple in edges_for_this_square:
            if all(edge in temp_edges_set for edge in edges_for_this_square):
                newly_completed_indices.append(idx)
    return newly_completed_indices
# --- End of GA Utility Functions ---


# --- GA Heuristic Component Functions (already in your provided code) ---
def count_potential_dangerous_moves(current_lines_drawn_tuples, squares_config, all_possible_edges_list, bot_make_moves):
    # current_edges_set = extract_edge_set(current_lines_drawn_tuples)
    # dangerous_move_count = 0
    # for edge_tuple in all_possible_edges_list:
    #     if edge_tuple not in current_edges_set:
    #         if is_dangerous_move(edge_tuple, current_lines_drawn_tuples, squares_config):
    #             dangerous_move_count += 1
    # return dangerous_move_count
    
    # dangerous_move_count = 0
    # temp_edges_set = set(current_lines_drawn_tuples)
    # for square_nodes in squares_config:
    #     edges_for_this_square = edges_of_square(square_nodes)
    #     count_filled_edges = sum(1 for edge in edges_for_this_square if edge in temp_edges_set)
    #     if count_filled_edges == 3:
    #         dangerous_move_count += 1
    # return dangerous_move_count
    
    current_edges_set = extract_edge_set(current_lines_drawn_tuples)

    if bot_make_moves == False:
        max_dangerous_count = 0
        for edge_tuple in all_possible_edges_list:
            if edge_tuple not in current_edges_set:
                chain_length_opp, _ = calculate_chain_length_from_move(edge_tuple, current_lines_drawn_tuples, squares_config, all_possible_edges_list)
                if max_dangerous_count < chain_length_opp:
                    max_dangerous_count = chain_length_opp
        return max_dangerous_count
    else:
        dangerous_count = 30
        for edge_tuple in all_possible_edges_list:
            if edge_tuple not in current_edges_set:
                new_lines_drawn_tuples = list(current_lines_drawn_tuples)
                new_lines_drawn_tuples.append(edge_tuple)
                potential_dangerous_count = count_potential_dangerous_moves(new_lines_drawn_tuples, squares_config, all_possible_edges_list, False)
                if dangerous_count > potential_dangerous_count:
                    dangerous_count = potential_dangerous_count
        if dangerous_count == 30:
            return 0
        return dangerous_count


def calculate_chain_length_from_move(potential_edge_tuple, current_lines_drawn_tuples, squares_config, all_possible_edges_list):
    sim_lines_drawn_tuples = list(current_lines_drawn_tuples)
    sim_edges_set = set(sim_lines_drawn_tuples)

    initially_completed_indices = check_new_squares(potential_edge_tuple, sim_lines_drawn_tuples, squares_config)
    if not initially_completed_indices:
        sim_lines_drawn_tuples.append(potential_edge_tuple)
        return 0, sim_lines_drawn_tuples

    chain_length = len(initially_completed_indices)
    sim_lines_drawn_tuples.append(potential_edge_tuple)
    sim_edges_set.add(potential_edge_tuple)
    completed_this_turn_indices = set(initially_completed_indices)

    while True:
        can_make_more_squares = False
        next_greedy_edge_tuple = None
        # In a full chain calculation, you'd find the best greedy edge.
        # For simplicity, taking the first one found.
        for edge_option_tuple in all_possible_edges_list:
            if edge_option_tuple not in sim_edges_set:
                newly_completed_by_option = check_new_squares(edge_option_tuple, sim_lines_drawn_tuples, squares_config, completed_this_turn_indices)
                if newly_completed_by_option:
                    next_greedy_edge_tuple = edge_option_tuple
                    chain_length += len(newly_completed_by_option)
                    completed_this_turn_indices.update(newly_completed_by_option)
                    can_make_more_squares = True
                    break
        if can_make_more_squares and next_greedy_edge_tuple:
            sim_lines_drawn_tuples.append(next_greedy_edge_tuple)
            sim_edges_set.add(next_greedy_edge_tuple)
        else:
            break
    # print(chain_length)
    return chain_length, sim_lines_drawn_tuples


# Giả định các hàm này đã được định nghĩa ở đâu đó:
# def extract_edge_set(lines_tuples):
#     return set(lines_tuples)
#
# def check_new_squares(edge_to_add, current_lines_tuples, squares_config):
#     # Trả về một list các ID của ô vuông mới được hoàn thành,
#     # hoặc list rỗng nếu không có ô nào được hoàn thành.
#     # Ví dụ: return [1, 2] nếu 2 ô vuông được hoàn thành.
#     pass

def simulate_opponent_chain(
    lines_at_start_of_opponent_turn_list,
    edges_at_start_of_opponent_turn_set,
    all_possible_edges_list,
    squares_config,
):
    """
    Mô phỏng chuỗi nước đi tham lam của đối thủ.

    Args:
        lines_at_start_of_opponent_turn_list (list): Danh sách các đường đã vẽ
            tại thời điểm bắt đầu lượt của đối thủ.
        edges_at_start_of_opponent_turn_set (set): Tập hợp các cạnh đã vẽ
            tại thời điểm bắt đầu lượt của đối thủ.
        all_possible_edges_list (list): Danh sách tất cả các cạnh có thể có trên bàn cờ.
        squares_config: Cấu hình các ô vuông (cần thiết cho check_new_squares_func).
        check_new_squares_func (function): Hàm để kiểm tra xem một cạnh có tạo ra ô vuông mới không.
            Hàm này nhận (edge_option, current_lines_tuple, squares_config)
            và trả về list các ô vuông mới được tạo.

    Returns:
        tuple: (chain_length_opponent, final_sim_lines_list, final_sim_edges_set)
            - chain_length_opponent (int): Số ô vuông đối thủ hoàn thành trong chuỗi.
            - final_sim_lines_list (list): Danh sách đường kẻ sau khi chuỗi của đối thủ kết thúc.
            - final_sim_edges_set (set): Tập hợp cạnh sau khi chuỗi của đối thủ kết thúc.
    """
    # Tạo bản sao để không làm thay đổi list/set gốc được truyền vào
    sim_lines_list_for_opponent_chain = list(lines_at_start_of_opponent_turn_list)
    sim_edges_set_for_opponent_chain = edges_at_start_of_opponent_turn_set.copy()

    opponent_chain_length = 0

    while True:
        can_make_more_squares_opponent = False
        next_greedy_edge_for_opponent = None

        for edge_option_opponent in all_possible_edges_list:
            if edge_option_opponent not in sim_edges_set_for_opponent_chain:
                # check_new_squares_func kiểm tra xem việc thêm edge_option_opponent
                # vào trạng thái hiện tại (sim_lines_list_for_opponent_chainc) có tạo ô vuông không.
                newly_completed_by_option = check_new_squares(
                    edge_option_opponent,
                    tuple(sim_lines_list_for_opponent_chain), # Chuyển thành tuple để đảm bảo tính bất biến nếu cần
                    squares_config
                )
                if newly_completed_by_option:
                    next_greedy_edge_for_opponent = edge_option_opponent
                    opponent_chain_length += len(newly_completed_by_option)
                    can_make_more_squares_opponent = True
                    break  # Đối thủ chọn nước đi tham lam đầu tiên tìm thấy

        if can_make_more_squares_opponent and next_greedy_edge_for_opponent:
            sim_lines_list_for_opponent_chain.append(next_greedy_edge_for_opponent)
            sim_edges_set_for_opponent_chain.add(next_greedy_edge_for_opponent)
        else:
            break  # Chuỗi của đối thủ kết thúc

    return opponent_chain_length, sim_lines_list_for_opponent_chain, sim_edges_set_for_opponent_chain


def count_potential_reward(
    current_lines_drawn_tuples,
    squares_config,
    all_possible_edges_list,
):
    """
    Tính toán "phần thưởng tiềm năng" dựa trên chuỗi ngắn nhất mà đối thủ có thể tạo.
    "Phần thưởng tiềm năng" ở đây thực ra là một "chi phí" hoặc "mối nguy hiểm" tiềm tàng,
    vì nó đo lường lợi ích của đối thủ. Giá trị càng nhỏ càng tốt cho chúng ta.

    Args:
        current_lines_drawn_tuples (tuple): Tuple các đường đã vẽ hiện tại.
        squares_config: Cấu hình các ô vuông.
        all_possible_edges_list (list): Danh sách tất cả các cạnh có thể.
        extract_edge_set_func (function): Hàm để trích xuất tập hợp các cạnh từ danh sách đường.
        check_new_squares_func (function): Hàm để kiểm tra ô vuông mới.

    Returns:
        int: Chiều dài chuỗi ngắn nhất mà đối thủ có thể tạo ra sau một nước đi giả định của chúng ta.
             Nếu không có nước đi nào của chúng ta dẫn đến việc đối thủ tạo chuỗi,
             và không có nước đi nào còn lại, giá trị có thể vẫn là 100 (hoặc giá trị khởi tạo).
    """
    potential_reward = 100  # Giả sử 100 là một giá trị đủ lớn (tệ nhất cho chúng ta)

    # Trạng thái ban đầu, không bị thay đổi bởi các mô phỏng
    initial_current_lines_list = list(current_lines_drawn_tuples)
    initial_current_edges_set = extract_edge_set(current_lines_drawn_tuples)

    chain_length, initial_current_lines_list, initial_current_edges_set = simulate_opponent_chain(
        initial_current_lines_list,
        initial_current_edges_set,
        all_possible_edges_list,
        squares_config,
    )

    # Duyệt qua tất cả các nước đi khả thi của "chúng ta"
    for edge_tuple_our_hypothetical_move in all_possible_edges_list:
        if edge_tuple_our_hypothetical_move not in initial_current_edges_set:
            # 1. Giả định "chúng ta" thực hiện nước đi này
            # Tạo trạng thái mới SAU KHI chúng ta đi nước 'edge_tuple_our_hypothetical_move'
            lines_after_our_move_list = list(initial_current_lines_list)
            lines_after_our_move_list.append(edge_tuple_our_hypothetical_move)

            edges_after_our_move_set = initial_current_edges_set.copy()
            edges_after_our_move_set.add(edge_tuple_our_hypothetical_move)

            # 2. Kiểm tra xem nước đi của chúng ta có tạo ô vuông nào không
            # Logic này thường được xử lý bởi người chơi hiện tại, không phải trong hàm đánh giá này.
            # Hàm này đang xem xét phản ứng của đối thủ.
            # Nếu nước đi của chúng ta tạo ô, chúng ta được đi tiếp (theo luật Dots and Boxes thông thường).
            # Tuy nhiên, hàm này đang tính "potential_reward" dưới góc độ đối thủ sẽ được lợi gì
            # *sau khi* chúng ta kết thúc lượt của mình (tức là chúng ta không tạo thêm ô nào từ nước đi này,
            # hoặc đã tạo xong và giờ đến lượt đối thủ).

            # 3. Mô phỏng chuỗi phản ứng của đối thủ từ trạng thái này
            chain_length_from_opponent, _final_lines, _final_edges = simulate_opponent_chain(
                lines_after_our_move_list,
                edges_after_our_move_set,
                all_possible_edges_list,
                squares_config,
            )

            # Chúng ta đang tìm nước đi của mình mà dẫn đến chuỗi *ngắn nhất* của đối thủ.
            # (Nghĩa là, chúng ta muốn đối thủ được lợi ít nhất)
            if chain_length_from_opponent - chain_length < potential_reward:
                potential_reward = chain_length_from_opponent - chain_length

    # Nếu không có nước đi nào khả thi cho chúng ta (ví dụ: tất cả các cạnh đã được vẽ),
    # vòng lặp for sẽ không chạy, và potential_reward sẽ giữ giá trị khởi tạo.
    # Hoặc nếu tất cả các nước đi của chúng ta đều dẫn đến chain_length_from_opponent >= 100.
    if potential_reward == 100:
        return 0
    return potential_reward


def calculate_score_diff_after_move(potential_edge_tuple, current_lines_drawn_tuples, squares_config,
                                    score_player, score_opponent):
    newly_completed = check_new_squares(potential_edge_tuple, current_lines_drawn_tuples, squares_config)
    score_player_after_move = score_player + len(newly_completed)
    return score_player_after_move - score_opponent
# --- End of GA Heuristic Component Functions ---


# --- GA Core Logic (already in your provided code, minor adjustments if any) ---
POPULATION_SIZE = 20
NUM_GENERATIONS = 30
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
TOURNAMENT_SIZE = 3
NUM_GAMES_PER_EVALUATION = 4
MIN_WEIGHT = -10.0
MAX_WEIGHT = 10.0

def calculate_heuristic_value(potential_edge_tuple, weights,
                              current_lines_drawn_tuples, squares_config, all_possible_edges_list,
                              current_score_bot, current_score_opponent):
    w1, w2, w3, w4 = weights
    lines_after_potential_move = list(current_lines_drawn_tuples)
    lines_after_potential_move.append(potential_edge_tuple)

    #  tính chuỗi tạo được nếu đi nước hiện tại (chưa quan tâm nước sau có bị sao ko:) ? )
    a2_val, sim_lines_tuples = calculate_chain_length_from_move(potential_edge_tuple, current_lines_drawn_tuples, squares_config, all_possible_edges_list)
    # print(len(sim_lines_tuples) - len(current_lines_drawn_tuples))

    # sau khi đi nước đi hiện tại xong 
    if a2_val:
        a1_val = count_potential_dangerous_moves(sim_lines_tuples, squares_config, all_possible_edges_list, True)
    else:
        a1_val = count_potential_dangerous_moves(sim_lines_tuples, squares_config, all_possible_edges_list, False)

    if a2_val:
        a4_val = 0
    else:
        a4_val = count_potential_reward(sim_lines_tuples, squares_config, all_possible_edges_list)
    
    # print(a1_val, a2_val, a4_val, potential_edge_tuple, sim_lines_tuples)
    num_direct_completed = len(check_new_squares(potential_edge_tuple, current_lines_drawn_tuples, squares_config))
    temp_score_bot = current_score_bot + num_direct_completed
    a3_val = temp_score_bot - current_score_opponent
    return w1 * a1_val + w2 * a2_val + w3 * a3_val + w4 * a4_val

def choose_move_for_ga_bot(weights, current_lines_drawn_tuples, squares_config, all_possible_edges_list,
                           current_score_bot, current_score_opponent):
    current_edges_set = extract_edge_set(current_lines_drawn_tuples)
    available_moves_tuples = [edge for edge in all_possible_edges_list if edge not in current_edges_set]
    if not available_moves_tuples: return None
    best_move_tuple = None
    max_heuristic_val = -float('inf')
    for move_tuple in available_moves_tuples:
        val = calculate_heuristic_value(move_tuple, weights, current_lines_drawn_tuples, squares_config, all_possible_edges_list, current_score_bot, current_score_opponent)
        if val > max_heuristic_val:
            max_heuristic_val = val
            best_move_tuple = move_tuple
    if best_move_tuple is None and available_moves_tuples: # Should not happen if available_moves is not empty
        best_move_tuple = random.choice(available_moves_tuples)
    return best_move_tuple




cntGame = 0

def simulate_game(ga_weights, mcts_bot_func, squares_config, all_possible_edges_list, ga_bot_goes_first=True):
    global cntGame
    cntGame += 1
    # print("---------------- Game: " + str(cntGame) + " ------------------ ")
    lines_drawn_sim_tuples = []
    score_ga = 0
    score_mcts = 0
    num_total_squares = len(squares_config)
    current_player_is_ga = ga_bot_goes_first

    list_moves = []

    while score_ga + score_mcts < num_total_squares:
        move_tuple = None
        if current_player_is_ga:
            move_tuple = choose_move_for_ga_bot(ga_weights, lines_drawn_sim_tuples, squares_config, all_possible_edges_list, score_ga, score_mcts)

            # a2_val, sim_lines_tuples = calculate_chain_length_from_move(move_tuple, lines_drawn_sim_tuples, squares_config, all_possible_edges_list)
            # a1_val = count_potential_dangerous_moves(sim_lines_tuples, squares_config, all_possible_edges_list)
            # num_direct_completed = len(check_new_squares(move_tuple, lines_drawn_sim_tuples, squares_config))
            # temp_score_bot = score_ga + num_direct_completed
            # a3_val = temp_score_bot - score_mcts
            # print("Check: ", end = " ")
            # print(a1_val, a2_val, a3_val)
        else:
            # MCTS bot (player 2) is called with its score first, then opponent's (GA bot) score
            move_tuple = mcts_bot_func(lines_drawn_sim_tuples, squares_config, all_possible_edges_list, score_mcts, score_ga, time_limit=0.1) # Default MCTS time_limit



        # if (current_player_is_ga):
        #     print("GA: ", end = " ")
        # else:
        #     print("MCTS: ", end = " ")
        # print(move_tuple)
        list_moves.append({"botGA": current_player_is_ga, "move": move_tuple})

        if move_tuple is None: break

        # Store state *before* adding the move to pass to check_new_squares
        lines_before_this_move = list(lines_drawn_sim_tuples)
        lines_drawn_sim_tuples.append(move_tuple)
        
        newly_completed_indices = check_new_squares(move_tuple, lines_before_this_move, squares_config)

        if newly_completed_indices:
            if current_player_is_ga:
                score_ga += len(newly_completed_indices)
            else:
                score_mcts += len(newly_completed_indices)
            # Current player gets another turn
        else:
            current_player_is_ga = not current_player_is_ga # Switch turn
    
    # if score_mcts > score_ga:
    #     for move in list_moves:
    #         print(move)
    #     print("-----------------------------------------------")

    # print("Result: " + str(score_ga) + " - " + str(score_mcts))
    writeln = "Result: " + str(score_ga) + " - " + str(score_mcts)
    with open("result.txt", "a") as file:
        file.write(writeln)



    if score_ga > score_mcts: return 1
    elif score_mcts > score_ga: return -1
    else: return 0

def calculate_fitness(individual_weights, mcts_bot_func, squares_config, all_possible_edges_list):
    total_score = 0
    W, D, L = 0, 0, 0

    for i in range(NUM_GAMES_PER_EVALUATION):
        ga_goes_first = (i % 2 == 0)
        game_result = simulate_game(individual_weights, mcts_bot_func, squares_config, all_possible_edges_list, ga_goes_first)

        if game_result == 1:
            W += 1
        elif game_result == 0:
            D += 1
        elif game_result == -1:
            L += 1

        total_score += game_result

    print(f"W: {W}, D: {D}, L: {L}")
    return total_score


def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = [
            random.uniform(MIN_WEIGHT, 0),       # gene 0
            random.uniform(0, 10),               # gene 1
            random.uniform(0, 10),               # gene 2
            random.uniform(5, 10)                # gene 3
        ]
        population.append(individual)
    return population

def selection(population, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [(fitness_scores[i], population[i]) for i in tournament_indices]
        winner = max(tournament_fitness, key=lambda x: x[0])[1]
        selected_parents.append(winner)
    return selected_parents

def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        child1 = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)]
        child2 = [(p1 + p2) / 2 for p1, p2 in zip(parent1, parent2)] # Simple average
        return child1, child2
    return list(parent1), list(parent2)

def mutate(individual, mutation_rate):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            change = random.uniform(-abs(mutated_individual[i]*0.2) - 0.5, abs(mutated_individual[i]*0.2) + 0.5)
            mutated_individual[i] += change
            mutated_individual[i] = max(MIN_WEIGHT, min(MAX_WEIGHT, mutated_individual[i]))
    return mutated_individual

def optimize_heuristic_weights_ga(squares_config, mcts_bot_to_play_against_func):
    print("Starting Genetic Algorithm to optimize heuristic weights...")
    all_possible_edges_list = list(set(edge for sq_nodes in squares_config for edge in edges_of_square(sq_nodes)))
    print(f"Board has {len(squares_config)} squares and {len(all_possible_edges_list)} possible edges.")

    population = initialize_population(POPULATION_SIZE)
    best_overall_weights = None
    best_overall_fitness = -float('inf')

    for generation in range(NUM_GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{NUM_GENERATIONS} ---")
        fitness_scores = [calculate_fitness(ind_weights, mcts_bot_to_play_against_func, squares_config, all_possible_edges_list) for ind_weights in population]
        
        current_best_fitness_idx = fitness_scores.index(max(fitness_scores))
        current_best_weights = population[current_best_fitness_idx]
        current_max_fitness = fitness_scores[current_best_fitness_idx]
        
        print(f"  Best fitness in generation {generation + 1}: {current_max_fitness}")
        print(f"  Best weights in generation: {[round(w, 2) for w in current_best_weights]}")

        if current_max_fitness > best_overall_fitness:
            best_overall_fitness = current_max_fitness
            best_overall_weights = list(current_best_weights)
            print(f"  *** New best overall fitness found: {best_overall_fitness} ***")
            print(f"  *** With weights: {[round(w, 2) for w in best_overall_weights]} ***")

        parents = selection(population, fitness_scores, TOURNAMENT_SIZE)
        next_population = []
        
        sorted_population_with_fitness = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        num_elites = int(0.1 * POPULATION_SIZE)
        for i in range(num_elites):
            next_population.append(list(sorted_population_with_fitness[i][0]))

        while len(next_population) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = crossover(p1, p2, CROSSOVER_RATE)
            next_population.append(mutate(child1, MUTATION_RATE))
            if len(next_population) < POPULATION_SIZE:
                next_population.append(mutate(child2, MUTATION_RATE))
        population = next_population

    print("\n--- GA Optimization Finished ---")
    if best_overall_weights:
        print(f"Best overall weights found: {[round(w, 3) for w in best_overall_weights]}")
    else:
        print("No best weights found (this can happen with few generations/evaluations).")
    print(f"With fitness: {best_overall_fitness} (based on {NUM_GAMES_PER_EVALUATION} games per evaluation)")
    return best_overall_weights
# --- End of GA Core Logic ---

# ---------------- ADAPTED BOT MCTS -----------------------------
def adapted_bot_choose_move_MCTS(current_lines_drawn_tuples, # GA: list of edge tuples
                                 squares_config,             # GA: list of square node tuples
                                 all_possible_edges_list,    # GA: list of all possible edge tuples
                                 current_mcts_score,         # MCTS bot's current score
                                 current_opponent_score,     # Opponent's current score
                                 time_limit=0.1):            # MCTS time limit (adjust as needed for GA speed)
    start_time = time.time()

    current_edges_set = set(current_lines_drawn_tuples)
    # 1. Generate valid possible moves from all_possible_edges_list
    possible_root_moves = [edge for edge in all_possible_edges_list if edge not in current_edges_set]

    if not possible_root_moves:
        return None

    # 2. Ưu tiên nước đi tạo thành hình vuông (Greedy check for immediate win)
    for move_tuple in possible_root_moves:
        if check_new_squares(move_tuple, current_lines_drawn_tuples, squares_config):
            return move_tuple # Take the winning move immediately

    # 3. Loại bỏ các nước nguy hiểm for root move consideration if safe options exist
    safe_root_moves = [
        m_tuple for m_tuple in possible_root_moves
        if not is_dangerous_move(m_tuple, current_lines_drawn_tuples, squares_config)
    ]

    # Moves MCTS will actually simulate from. If safe moves exist, use them. Otherwise, use all.
    simulation_candidate_moves = safe_root_moves if safe_root_moves else possible_root_moves
    
    if not simulation_candidate_moves: # Should only happen if possible_root_moves was empty already
        return None

    move_stats = {move_tuple: {"wins": 0, "sims": 0} for move_tuple in simulation_candidate_moves}

    # Inner function for MCTS playout
    def mcts_playout(first_move_by_mcts_tuple, initial_board_tuples,
                     initial_score_mcts, initial_score_opponent):
        
        sim_lines_tuples = list(initial_board_tuples) # Board state *before* MCTS's first move
        sim_score_mcts = initial_score_mcts
        sim_score_opponent = initial_score_opponent
        
        current_player_in_sim = 2 # Player 2 is MCTS, Player 1 is Opponent

        # --- MCTS makes its first move in this simulation ('first_move_by_mcts_tuple') ---
        newly_completed_by_first_move = check_new_squares(first_move_by_mcts_tuple, sim_lines_tuples, squares_config)
        sim_lines_tuples.append(first_move_by_mcts_tuple) # Add the move to the board

        if newly_completed_by_first_move:
            sim_score_mcts += len(newly_completed_by_first_move)
            # MCTS (player 2) continues turn
        else:
            current_player_in_sim = 1 # Switch to Opponent (player 1)

        # --- Continue Playout Loop ---
        num_total_squares = len(squares_config)
        # Max playout length to prevent infinite loops in rare cases
        max_additional_moves_in_playout = len(all_possible_edges_list) - len(sim_lines_tuples) 

        for _ in range(max_additional_moves_in_playout):
            if sim_score_mcts + sim_score_opponent >= num_total_squares:
                break # Game over

            # Get available moves for current player in sim
            playout_current_edges_set = set(sim_lines_tuples)
            playout_available_moves = [
                edge for edge in all_possible_edges_list if edge not in playout_current_edges_set
            ]

            if not playout_available_moves:
                break # No more moves

            # Playout policy: 1. Complete square, 2. Safe random, 3. Any random
            next_playout_move_tuple = None
            # 1. Try to complete a square
            for p_move in playout_available_moves:
                if check_new_squares(p_move, sim_lines_tuples, squares_config):
                    next_playout_move_tuple = p_move
                    break
            
            if next_playout_move_tuple is None:
                # 2. Try to make a random safe move
                playout_safe_moves = [
                    p_move for p_move in playout_available_moves
                    if not is_dangerous_move(p_move, sim_lines_tuples, squares_config)
                ]
                if playout_safe_moves:
                    next_playout_move_tuple = random.choice(playout_safe_moves)
                else:
                    # 3. Make any random valid move
                    next_playout_move_tuple = random.choice(playout_available_moves)

            # Execute the chosen playout move
            lines_before_playout_move = list(sim_lines_tuples) # State before this specific playout move
            sim_lines_tuples.append(next_playout_move_tuple)
            newly_completed_in_playout = check_new_squares(next_playout_move_tuple, lines_before_playout_move, squares_config)

            if newly_completed_in_playout:
                if current_player_in_sim == 1: # Opponent
                    sim_score_opponent += len(newly_completed_in_playout)
                else: # MCTS
                    sim_score_mcts += len(newly_completed_in_playout)
                # Current player continues turn
            else:
                current_player_in_sim = 3 - current_player_in_sim # Switch turn (1->2, 2->1)
        
        # Determine winner of this playout
        if sim_score_mcts > sim_score_opponent: return 2 # MCTS wins
        elif sim_score_opponent > sim_score_mcts: return 1 # Opponent wins
        else: return 0 # Draw

    # 4. MCTS Simulation Loop
    num_sims_run = 0
    # Heuristic limit on simulations, useful if time_limit is generous for small boards
    max_sims_total = len(simulation_candidate_moves) * 150 if simulation_candidate_moves else 100 

    while (time.time() - start_time < time_limit) and (num_sims_run < max_sims_total) :
        if not simulation_candidate_moves: break

        # Select a root move to simulate from (simple random choice among candidates)
        # For a full MCTS, this would involve UCB1 tree traversal for selection.
        root_move_to_sim = random.choice(simulation_candidate_moves)

        playout_winner = mcts_playout(root_move_to_sim, current_lines_drawn_tuples,
                                      current_mcts_score, current_opponent_score)
        
        move_stats[root_move_to_sim]["sims"] += 1
        if playout_winner == 2: # MCTS (Player 2) won the playout
            move_stats[root_move_to_sim]["wins"] += 1
        num_sims_run +=1

    # 5. Choose the best move based on simulation stats (e.g., highest win rate)
    if not simulation_candidate_moves: # Should be caught earlier
        return random.choice(possible_root_moves) if possible_root_moves else None

    best_move_tuple = max(
        simulation_candidate_moves,
        key=lambda m_tuple: (move_stats[m_tuple]["wins"] / (move_stats[m_tuple]["sims"] + 1e-6)) if move_stats[m_tuple]["sims"] > 0 else -1.0
    )
    return best_move_tuple
# --------------------------------------------------

def generate_squares_for_grid(rows, cols):
    squares = []
    points_per_row = cols + 1
    for r in range(rows):
        for c in range(cols):
            p1 = r * points_per_row + c; p2 = p1 + 1
            p3 = (r + 1) * points_per_row + c; p4 = p3 + 1
            squares.append(tuple(sorted((p1, p2, p3, p4))))
    return squares

squares_config_4x4 = generate_squares_for_grid(4, 4)


# ------------------------------------------------------------------------------
def bot_choose_move(lines_drawn, squares, list_squares_1 = [], list_squares_2 = []):
    lines_drawn_tuples = [(item["id1"], item["id2"]) for item in lines_drawn]
    all_possible_edges_list = list(set(edge for sq_nodes in squares for edge in edges_of_square(sq_nodes)))    
    weights = [-4.0, 6.53, 5.62, 7.48]
    #  [-4.0, 6.53, 5.62, 7.48]
    return choose_move_for_ga_bot(weights, lines_drawn_tuples, squares, all_possible_edges_list, len(list_squares_1), len(list_squares_2))

# ------------------------------------------------------------------------------


if __name__ == '__main__':
    print(f"Generated 4x4 squares_config: {squares_config_4x4[:3]}... ({len(squares_config_4x4)} squares)")
    
    # To test MCTS bot standalone (Optional)
    # print("\n--- Standalone MCTS Test ---")
    # test_lines = [] # Empty board
    # test_all_edges = list(set(edge for sq_nodes in squares_config_4x4 for edge in edges_of_square(sq_nodes)))
    # test_mcts_score = 0
    # test_opponent_score = 0
    # for i in range(3):
    #     print(f"MCTS Turn {i+1}, board: {test_lines}")
    #     chosen_mcts_move = adapted_bot_choose_move_MCTS(
    #         test_lines, squares_config_4x4, test_all_edges,
    #         test_mcts_score, test_opponent_score, time_limit=0.5
    #     )
    #     print(f"MCTS Chose: {chosen_mcts_move}")
    #     if chosen_mcts_move:
    #         lines_before = list(test_lines)
    #         test_lines.append(chosen_mcts_move)
    #         completed = check_new_squares(chosen_mcts_move, lines_before, squares_config_4x4)
    #         if completed:
    #             test_mcts_score += len(completed)
    #             print(f"MCTS scored {len(completed)} boxes. New MCTS score: {test_mcts_score}")
    #         # In a real game, turn would switch if no boxes completed
    #     else:
    #         print("MCTS found no move.")
    #         break
    # print("--- End Standalone MCTS Test ---\n")

    # --- GA Optimization ---
    # Reduce parameters for quicker testing if needed
    # POPULATION_SIZE = 10
    # NUM_GENERATIONS = 5
    # NUM_GAMES_PER_EVALUATION = 2
    # For `adapted_bot_choose_move_MCTS`, ensure the time_limit in `simulate_game` call is appropriate.
    # If it's too long, GA will be very slow. If too short, MCTS plays poorly.
    # The default in `simulate_game` is 0.1s.

    optimized_weights_4x4 = optimize_heuristic_weights_ga(squares_config_4x4, adapted_bot_choose_move_MCTS)
    
    if optimized_weights_4x4:
        print("\nExample of how to use the optimized weights in your game loop:")
        print("best_move = choose_move_for_ga_bot(optimized_weights_4x4, current_lines_drawn, squares_config_4x4, all_possible_edges_list, my_score, opponent_score)")