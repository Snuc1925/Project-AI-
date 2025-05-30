import pygame
import sys
from pygame import gfxdraw
import os
import random
import math
import time

# Colors and game settings
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_RED = (255, 114, 114)
BLUE = (0, 0, 255)
LIGHT_BLUE = (148, 249, 237)
GREEN = (0, 200, 0)
DOT_RADIUS = 5
GRID_SPACING = 100
MARGIN = 100
CLICK_RADIUS = 15
LINE_COLOR = BLACK
DOT_HIGHLIGHT_RADIUS = 7
THICKNESS = 3

# Heuristic constants
COMPLETE_SQUARE_VALUE = 15.0
THREE_EDGE_VALUE = 9.0
TWO_EDGE_VALUE = 2.5
ONE_EDGE_VALUE = 1.0
CENTER_CONTROL_VALUE = 4.5
CHAIN_PENALTY_BASE = 18.0
OPPONENT_THREE_EDGE_PENALTY = -10.0
CHAIN_DISRUPTION_BONUS = 4.0
DOUBLE_CROSS_BONUS = 15.0
CHAIN_LENGTH_MULTIPLIER = 1.5
FIRST_MOVE_BONUS = 3.0
CHAIN_CREATION_BONUS = 5.0

# Time management constants
MIN_MOVE_TIME = 1.5
MAX_MOVE_TIME = 7.5
BASE_MOVE_TIME = 4.5
CRITICAL_MOVE_TIME = 7.0

# Transposition Table
transposition_table = {}  # {hash_key: (score, depth, move)}
TABLE_SIZE = 1000000

# Zobrist Hashing
def initialize_zobrist_table(num_dots):
    zobrist_table = {}
    for i in range(num_dots):
        for j in range(i + 1, num_dots):
            zobrist_table[(i, j)] = random.getrandbits(64)
    return zobrist_table

def compute_zobrist_hash(lines_drawn, zobrist_table):
    hash_value = 0
    for line in lines_drawn:
        edge = tuple(sorted((line['id1'], line['id2'])))
        if edge in zobrist_table:
            hash_value ^= zobrist_table[edge]
    return hash_value

def store_transposition(hash_key, score, depth, move):
    if len(transposition_table) < TABLE_SIZE:
        transposition_table[hash_key] = (score, depth, move)
    elif depth > transposition_table.get(hash_key, (0, -1, None))[1]:
        transposition_table[hash_key] = (score, depth, move)

def lookup_transposition(hash_key, depth):
    entry = transposition_table.get(hash_key)
    if entry and entry[1] >= depth:
        return entry[0], entry[2]
    return None, None

# MCTS Node
class MCTSNode:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_moves = None

    def select_child(self, c=0.4):
        if not self.children:
            return None
        visits = sum(child.visits for child in self.children)
        k = max(3, int(math.sqrt(visits) * 1.5))
        top_k = sorted(self.children, 
                       key=lambda child: child.value / child.visits if child.visits > 0 else float('inf'),
                       reverse=True)[:k]
        return max(top_k, key=lambda child: 
                   (child.value / child.visits if child.visits > 0 else float('inf')) + 
                   c * math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1)))

    def expand(self, move):
        child = MCTSNode(move=move, parent=self)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child

    def update(self, result):
        self.visits += 1
        self.value += result

# Utility functions
def check_new_squares(new_edge, lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(new_edge)
    new_squares = []
    for idx, square in enumerate(squares):
        edges = edges_of_square(square)
        if new_edge in edges:
            if all(edge in current_edges for edge in edges):
                new_squares.append(idx)
    return new_squares

def edges_of_square(square):
    a, b, c, d = square
    return [tuple(sorted(edge)) for edge in [(a, b), (a, c), (b, d), (c, d)]]

def extract_edge_set(lines_drawn):
    return set(tuple(sorted((line['id1'], line['id2']))) for line in lines_drawn)

def get_valid_moves(lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    valid_moves = set()
    for square in squares:
        edges = edges_of_square(square)
        for edge in edges:
            if edge not in current_edges:
                valid_moves.add(edge)
    return list(valid_moves)

def is_game_over(lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            return False
    return True

def get_board_size(squares):
    if not squares:
        return (0, 0)
    max_row = max(max(square) for square in squares) // (len(squares) + 1)
    max_col = max(max(square) for square in squares) % (len(squares) + 1)
    return (max_row + 1, max_col + 1)

def get_center_squares(squares):
    rows, cols = get_board_size(squares)
    center_row = rows // 2
    center_col = cols // 2
    center_squares = []
    for square in squares:
        a, b, c, d = square
        row = a // cols
        col = a % cols
        if abs(row - center_row) <= 1 and abs(col - center_col) <= 1:
            center_squares.append(square)
    return center_squares

def is_square_controlled(square, edges):
    square_edges = edges_of_square(square)
    return all(edge in edges for edge in square_edges)

def get_opening_move(board_size, lines_drawn):
    opening_strategy = {
        '3x3': {
            'first_move': [(0,1), (0,3), (1,4), (3,6), (4,7)],
            'responses': {
                (0,1): [(1,4), (0,3), (3,6)],
                (0,3): [(1,4), (0,1), (4,7)],
                (1,4): [(0,1), (0,3), (3,6)],
                (3,6): [(0,1), (1,4), (4,7)],
                (4,7): [(0,3), (1,4), (3,6)]
            }
        },
        '4x4': {
            'first_move': [(0,1), (0,4), (1,5), (4,8), (5,9)],
            'responses': {
                (0,1): [(1,5), (0,4), (4,8)],
                (0,4): [(1,5), (0,1), (5,9)],
                (1,5): [(0,1), (0,4), (4,8)],
                (4,8): [(0,1), (1,5), (5,9)],
                (5,9): [(0,4), (1,5), (4,8)]
            }
        },
        '5x5': {
            'first_move': [(0,1), (0,5), (1,6), (5,11), (6,12)],
            'responses': {
                (0,1): [(1,6), (0,5), (5,11)],
                (0,5): [(1,6), (0,1), (6,12)],
                (1,6): [(0,1), (0,5), (5,11)],
                (5,11): [(0,1), (1,6), (6,12)],
                (6,12): [(0,5), (1,6), (5,11)]
            }
        },
        '4x5': {
            'first_move': [(0,1), (0,4), (1,5), (4,9), (5,10)],
            'responses': {
                (0,1): [(1,5), (0,4), (4,9)],
                (0,4): [(1,5), (0,1), (5,10)],
                (1,5): [(0,1), (0,4), (4,9)],
                (4,9): [(0,1), (1,5), (5,10)],
                (5,10): [(0,4), (1,5), (4,9)]
            }
        }
    }
    
    key = f"{board_size[0]}x{board_size[1]}"
    if key not in opening_strategy:
        return None
        
    if len(lines_drawn) == 0:
        return random.choice(opening_strategy[key]['first_move'])
    elif len(lines_drawn) == 1:
        last_move = tuple(sorted((lines_drawn[-1]['id1'], lines_drawn[-1]['id2'])))
        if last_move in opening_strategy[key]['responses']:
            return random.choice(opening_strategy[key]['responses'][last_move])
    
    return None

# Heuristic functions
def evaluate_position(lines_drawn, squares):
    score = 0
    current_edges = extract_edge_set(lines_drawn)
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0
    complete_square_value = 15.0 if game_phase > 0.7 else 12.0

    if len(lines_drawn) < 4:
        score += 3.0 * (1 - game_phase)
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            score += THREE_EDGE_VALUE * (1 + game_phase * 0.5)
        elif len(existing) == 2:
            score += TWO_EDGE_VALUE * (1 - game_phase * 0.3)
        elif len(existing) == 1:
            score += ONE_EDGE_VALUE

    center_squares = get_center_squares(squares)
    for square in center_squares:
        if is_square_controlled(square, current_edges):
            score += CENTER_CONTROL_VALUE * (1 - game_phase * 0.5)

    three_edge_count = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
            missing_edge = [edge for edge in edges if edge not in current_edges][0]
            temp_lines = lines_drawn.copy()
            temp_lines.append({"id1": missing_edge[0], "id2": missing_edge[1], "player": 2})
            if len(check_new_squares(missing_edge, temp_lines, squares)) > 0:
                score += 5.0 * (1 - game_phase * 0.5)

    opponent_three_edge = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            missing_edge = [edge for edge in edges if edge not in current_edges][0]
            temp_lines = lines_drawn + [{"id1": missing_edge[0], "id2": missing_edge[1], "player": 1}]
            if check_new_squares(missing_edge, temp_lines, squares):
                opponent_three_edge += 1
    score += opponent_three_edge * OPPONENT_THREE_EDGE_PENALTY * (1 + game_phase)

    return score

def find_forced_moves(lines_drawn, squares):
    forced_moves = []
    current_edges = extract_edge_set(lines_drawn)
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            missing = [edge for edge in edges if edge not in current_edges]
            if missing:
                forced_moves.append(missing[0])
    return forced_moves

def predict_chain_reaction(move, lines_drawn, squares, start_time):
    bot_boxes = 0
    opponent_boxes = 0
    temp_lines = lines_drawn.copy()
    temp_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    bot_boxes += len(check_new_squares(move, lines_drawn, squares))
    current_edges = extract_edge_set(temp_lines)

    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0

    three_edge_count = 0
    opponent_three_edge_count = 0
    favorable_chain_count = 0
    chain_length = 0
    max_chain_depth = 5
    
    chain_starts = set()
    chain_ends = set()
    
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
            missing_edge = [e for e in edges if e not in current_edges][0]
            temp_lines_opponent = temp_lines.copy()
            temp_lines_opponent.append({"id1": missing_edge[0], "id2": missing_edge[1], "player": 1})
            if len(check_new_squares(missing_edge, temp_lines_opponent, squares)) > 0:
                opponent_three_edge_count += 1
                chain_starts.add(missing_edge)
            else:
                favorable_chain_count += 1
                chain_ends.add(missing_edge)

    depth = 0
    while depth < max_chain_depth:
        forced_moves = find_forced_moves(temp_lines, squares)
        if not forced_moves:
            break
        chain_length += 1
        for forced_move in forced_moves:
            temp_lines.append({"id1": forced_move[0], "id2": forced_move[1], "player": 1})
            opponent_boxes += len(check_new_squares(forced_move, temp_lines, squares))
            current_edges = extract_edge_set(temp_lines)
        forced_moves = find_forced_moves(temp_lines, squares)
        if not forced_moves:
            break
        for forced_move in forced_moves:
            temp_lines.append({"id1": forced_move[0], "id2": forced_move[1], "player": 2})
            bot_boxes += len(check_new_squares(forced_move, temp_lines, squares))
            current_edges = extract_edge_set(temp_lines)
        depth += 1
        if time.time() - start_time > 4.5:
            break

    chain_structure_score = 0
    if chain_starts and chain_ends:
        chain_structure_score = len(chain_ends) * CHAIN_CREATION_BONUS
    elif chain_starts:
        chain_structure_score = -len(chain_starts) * CHAIN_PENALTY_BASE

    chain_structure_penalty = (three_edge_count * (1 + game_phase * 0.5) * CHAIN_PENALTY_BASE * 
                             (1 + chain_length * CHAIN_LENGTH_MULTIPLIER))
    opponent_chain_penalty = opponent_three_edge_count * CHAIN_PENALTY_BASE * (1 + game_phase)
    favorable_chain_bonus = favorable_chain_count * CHAIN_CREATION_BONUS * (1 - game_phase * 0.5)
    
    disruption_bonus = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 2:
            disruption_bonus += CHAIN_DISRUPTION_BONUS * (1 - game_phase * 0.3)

    return (bot_boxes - opponent_boxes - chain_structure_penalty - opponent_chain_penalty + 
            disruption_bonus + favorable_chain_bonus + chain_structure_score)

def evaluate_forced_move(move, lines_drawn, squares):
    new_lines = lines_drawn.copy()
    new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    completed_boxes = len(check_new_squares((move[0], move[1]), lines_drawn, squares))
    
    score = evaluate_position(new_lines, squares)
    
    three_edge_squares = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in extract_edge_set(new_lines)]
        if len(existing) == 3:
            three_edge_squares += 1
    double_cross_bonus = DOUBLE_CROSS_BONUS if three_edge_squares >= 2 else 0

    score += completed_boxes * COMPLETE_SQUARE_VALUE + double_cross_bonus
    return score

def move_priority(move, lines_drawn, squares):
    new_lines = lines_drawn.copy()
    new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    score = len(check_new_squares(move, lines_drawn, squares)) * COMPLETE_SQUARE_VALUE
    current_edges = extract_edge_set(new_lines)
    
    three_edge_count = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
    score += three_edge_count * THREE_EDGE_VALUE
    
    center_squares = get_center_squares(squares)
    center_edges = set()
    for square in center_squares:
        center_edges.update(edges_of_square(square))
    if move in center_edges:
        score += CENTER_CONTROL_VALUE * 0.5
    
    disruption_score = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 2 and move in edges:
            disruption_score += CHAIN_DISRUPTION_BONUS
    score += disruption_score
    
    return score

# MCTS Simulation Policy
def simulation_policy(lines_drawn, squares, current_edges):
    forced_moves = find_forced_moves(lines_drawn, squares)
    if forced_moves:
        scored_forced_moves = [(move, evaluate_forced_move(move, lines_drawn, squares)) 
                              for move in forced_moves]
        scored_forced_moves.sort(key=lambda x: x[1], reverse=True)
        return scored_forced_moves[0][0]
    
    valid_moves = get_valid_moves(lines_drawn, squares)
    if not valid_moves:
        return None
    
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0
    
    safe_moves = []
    risky_moves = []
    for move in valid_moves:
        new_lines = lines_drawn + [{"id1": move[0], "id2": move[1], "player": 2}]
        new_edges = extract_edge_set(new_lines)
        creates_opponent_three_edge = False
        for square in squares:
            edges = edges_of_square(square)
            existing = [e for e in edges if e in new_edges]
            if len(existing) == 3:
                creates_opponent_three_edge = True
                break
        if not creates_opponent_three_edge:
            safe_moves.append((move, move_priority(move, lines_drawn, squares)))
        else:
            risky_moves.append((move, move_priority(move, lines_drawn, squares)))
    
    scored_moves = safe_moves if safe_moves else risky_moves
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    k = max(3, len(scored_moves) // 3)
    top_moves = scored_moves[:k]
    return random.choice(top_moves)[0] if top_moves else random.choice(valid_moves)

# MCTS Simulation
def mcts_simulation(lines_drawn, squares, zobrist_table, start_time, time_limit):
    temp_lines = lines_drawn.copy()
    current_edges = extract_edge_set(temp_lines)
    current_player = 2
    
    chain_count = 0
    three_edge_squares = 0
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0
    
    while not is_game_over(temp_lines, squares):
        if time.time() - start_time > time_limit:
            break
            
        move = simulation_policy(temp_lines, squares, current_edges)
        if not move:
            break
            
        temp_lines.append({"id1": move[0], "id2": move[1], "player": current_player})
        new_squares = check_new_squares(move, temp_lines, squares)
        
        if new_squares:
            chain_count += len(new_squares)
        else:
            current_player = 3 - current_player
            
        current_edges = extract_edge_set(temp_lines)
        three_edge_squares = sum(1 for square in squares 
                               if len([e for e in edges_of_square(square) if e in current_edges]) == 3)
        completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
        game_phase = completed_squares / total_squares if total_squares > 0 else 0
    
    hash_key = compute_zobrist_hash(temp_lines, zobrist_table)
    cached_score, _ = lookup_transposition(hash_key, 0)
    if cached_score is not None:
        score = cached_score
    else:
        score = evaluate_position(temp_lines, squares)
        store_transposition(hash_key, score, 0, None)
    
    if game_phase > 0.7:
        score += chain_count * 1.5
        score -= three_edge_squares * 2.0
    else:
        score += chain_count * 2.5
        score -= three_edge_squares * 1.0
    
    return score

# MCTS Main Function
def mcts_search(lines_drawn, squares, zobrist_table, time_limit, start_time):
    root = MCTSNode()
    root.untried_moves = get_valid_moves(lines_drawn, squares)
    
    root.untried_moves.sort(key=lambda m: move_priority(m, lines_drawn, squares), reverse=True)
    
    iterations = 0
    max_iterations = 10000
    exploration_constant = 0.4

    while time.time() - start_time < time_limit and iterations < max_iterations:
        node = root
        temp_lines = lines_drawn.copy()
        current_player = 2

        while node.untried_moves is None and node.children:
            node = node.select_child(exploration_constant)
            temp_lines.append({"id1": node.move[0], "id2": node.move[1], "player": current_player})
            current_player = 3 - current_player

        if node.untried_moves:
            if len(node.untried_moves) > 3:
                node.untried_moves = node.untried_moves[:max(3, len(node.untried_moves) // 3)]
            move = random.choice(node.untried_moves)
            temp_lines.append({"id1": move[0], "id2": move[1], "player": current_player})
            node = node.expand(move)
            current_player = 3 - current_player

        result = mcts_simulation(temp_lines, squares, zobrist_table, start_time, time_limit)

        while node:
            node.update(result)
            node = node.parent

        iterations += 1

    if root.children:
        best_child = max(root.children, 
                        key=lambda c: (c.value / c.visits if c.visits > 0 else float('-inf')))
        return best_child.move
    elif root.untried_moves:
        return max(root.untried_moves, key=lambda m: move_priority(m, lines_drawn, squares))
    return None

# Bot Move Selection
def bot_choose_move(lines_drawn, squares):
    start_time = time.time()
    
    board_size = get_board_size(squares)
    num_dots = board_size[0] * board_size[1]
    zobrist_table = initialize_zobrist_table(num_dots)
    
    current_edges = extract_edge_set(lines_drawn)
    three_edge_count = sum(1 for square in squares 
                          if len([e for e in edges_of_square(square) if e in current_edges]) == 3)
    
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0
    
    if game_phase > 0.7 or three_edge_count >= 2:
        time_limit = CRITICAL_MOVE_TIME
    elif three_edge_count == 1:
        time_limit = BASE_MOVE_TIME + 1.0
    else:
        time_limit = BASE_MOVE_TIME

    mcts_time_limit = min(time_limit, MAX_MOVE_TIME - 0.5)
    
    opening_move = get_opening_move(board_size, lines_drawn)
    if opening_move:
        if len(lines_drawn) < 2:
            time.sleep(MIN_MOVE_TIME)
        return opening_move

    valid_moves = get_valid_moves(lines_drawn, squares)
    if not valid_moves:
        return None

    best_move = mcts_search(lines_drawn, squares, zobrist_table, mcts_time_limit, start_time)
    
    elapsed_time = time.time() - start_time
    if elapsed_time < MIN_MOVE_TIME:
        time.sleep(MIN_MOVE_TIME - elapsed_time)

    return best_move

# Graphics Functions
def draw_colored_squares(idx, squares, dot_positions, canvas, color=BLACK):
    a, b, c, d = squares[idx]
    points = [dot_positions[a], dot_positions[b], dot_positions[d], dot_positions[c]]
    pygame.draw.polygon(canvas, color, points)

def get_point_at_vt(dot_positions, x, y, click_radius=CLICK_RADIUS):
    for i, vt in enumerate(dot_positions):
        px, py = vt
        dist = (px - x)**2 + (py - y)**2
        if dist < click_radius**2:
            return i
    return None

def draw_line(surface, dot_positions, id1, id2, color, thickness=THICKNESS):
    num_dots = len(dot_positions)
    if not (0 <= id1 < num_dots and 0 <= id2 < num_dots): return False
    try:
        vt1 = dot_positions[id1]
        vt2 = dot_positions[id2]
        pygame.draw.line(surface, color, vt1, vt2, thickness)
        return True
    except IndexError:
        print(f"Error drawing line: Index out of range for IDs ({id1}, {id2})")
        return False

def generate_squares(rows, cols):
    squares = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            top_left = r * cols + c
            top_right = top_left + 1
            bottom_left = top_left + cols
            bottom_right = bottom_left + 1
            squares.append((top_left, top_right, bottom_left, bottom_right))
    return squares

def display_dots(rows, cols, mode):
    pygame.init()
    screen_width = (cols - 1) * GRID_SPACING + 2 * MARGIN
    screen_height = (rows - 1) * GRID_SPACING + 2 * MARGIN
    screen_width = max(screen_width, MARGIN * 2)
    screen_height = max(screen_height, MARGIN * 2)

    try:
        SURF = pygame.display.set_mode((screen_width, screen_height))
    except pygame.error as e:
        print(f"Error setting display mode ({screen_width}x{screen_height}): {e}")
        pygame.quit()
        return 
    
    pygame.display.set_caption("Dots and Boxes")
    
    icon_path = os.path.join(os.path.dirname(__file__), 'images', 'dotsandboxes.png')
    try:
        if os.path.exists(icon_path):
            icon_surf = pygame.image.load(icon_path)
            pygame.display.set_icon(icon_surf)
        else:
            print(f"Warning: Icon file not found at {icon_path}")
    except Exception as e:
        print(f"Warning: Could not load icon: {e}")

    try:
        score_font = pygame.font.SysFont('Arial', 30)
    except:
        score_font = pygame.font.Font(None, 40)

    if mode == "Person vs AI":
        player1_name = "Person"
        player2_name = "AI"
        player1_color = BLUE
        player2_color = RED
    elif mode == "AI vs AI":
        player1_name = "AI1"
        player2_name = "AI2"
        player1_color = BLUE
        player2_color = RED
    else:
        player1_name = "P1"
        player2_name = "P2"
        player1_color = BLUE
        player2_color = RED
        
    score_p1_text = f"{player1_name}: 0"
    score_p2_text = f"{player2_name}: 0"
    score_p1_surf = score_font.render(score_p1_text, True, player1_color)
    score_p2_surf = score_font.render(score_p2_text, True, player2_color)

    p1_width, p1_height = score_p1_surf.get_size()
    p2_width, p2_height = score_p2_surf.get_size()
    spacing = 50
    total_width = p1_width + spacing + p2_width
    start_x_p1 = (screen_width - total_width) // 2
    start_x_p2 = start_x_p1 + p1_width + spacing
    text_y = 30
    clock = pygame.time.Clock()

    dot_positions = []
    for r in range(rows):
        for c in range(cols):
            x = c * GRID_SPACING + MARGIN
            y = r * GRID_SPACING + MARGIN
            dot_positions.append((x, y))
    
    selected_dot_id = None
    lines_drawn = []
    current_player = 1
    squares = generate_squares(rows, cols)
    list_squares_1 = []
    list_squares_2 = []
            
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and current_player == 1 and mode != "AI vs AI":
                mouse_x, mouse_y = event.pos
                clicked_dot_id = get_point_at_vt(dot_positions, mouse_x, mouse_y)

                if clicked_dot_id is not None:
                    if selected_dot_id is None:
                        selected_dot_id = clicked_dot_id
                    elif selected_dot_id == clicked_dot_id:
                        selected_dot_id = None
                    else:
                        id1 = selected_dot_id
                        id2 = clicked_dot_id
                        is_adjacent = False
                        vt1 = dot_positions[id1]
                        vt2 = dot_positions[id2]

                        if (abs(vt1[1] - vt2[1]) < 5 and abs(vt1[0] - vt2[0] - GRID_SPACING) < 5) or \
                           (abs(vt1[1] - vt2[1]) < 5 and abs(vt1[0] - vt2[0] + GRID_SPACING) < 5) or \
                           (abs(vt1[0] - vt2[0]) < 5 and abs(vt1[1] - vt2[1] - GRID_SPACING) < 5) or \
                           (abs(vt1[0] - vt2[0]) < 5 and abs(vt1[1] - vt2[1] + GRID_SPACING) < 5):
                            is_adjacent = True

                        if is_adjacent:
                            id_pair = tuple(sorted((id1, id2)))
                            line_exists = any(line['id1'] == id_pair[0] and line['id2'] == id_pair[1] for line in lines_drawn)

                            if not line_exists:
                                new_squares_indices = check_new_squares(id_pair, lines_drawn, squares)
                                lines_drawn.append({"id1": id_pair[0], "id2": id_pair[1], "player": current_player})

                                if new_squares_indices:
                                    list_squares_1.extend(new_squares_indices)
                                else:
                                    current_player = 2
                                    
                                SURF.fill(WHITE)
                                score_p1_text = f"{player1_name}: {len(list_squares_1)}"
                                score_p2_text = f"{player2_name}: {len(list_squares_2)}"
                                score_p1_surf = score_font.render(score_p1_text, True, player1_color)
                                score_p2_surf = score_font.render(score_p2_text, True, player2_color)
                                SURF.blit(score_p1_surf, (start_x_p1, text_y))
                                SURF.blit(score_p2_surf, (start_x_p2, text_y))
                                for idx in list_squares_1:
                                    draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_BLUE)
                                for idx in list_squares_2:
                                    draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_RED)
                                for line_info in lines_drawn:
                                    draw_color = BLUE if line_info["player"] == 1 else RED
                                    draw_line(SURF, dot_positions, line_info["id1"], line_info["id2"], draw_color)
                                for i, pos in enumerate(dot_positions):
                                    x, y = pos
                                    radius = DOT_RADIUS
                                    dot_color = BLACK
                                    if i == selected_dot_id:
                                        radius = DOT_HIGHLIGHT_RADIUS
                                        dot_color = GREEN
                                    gfxdraw.filled_circle(SURF, x, y, radius, dot_color)
                                    gfxdraw.aacircle(SURF, x, y, radius, dot_color)
                                pygame.display.update()
                        selected_dot_id = None

        if current_player == 2 or (current_player == 1 and mode == "AI vs AI"):
            move = bot_choose_move(lines_drawn, squares)
            if move:
                id1, id2 = move
                new_squares_indices = check_new_squares((id1, id2), lines_drawn, squares)
                lines_drawn.append({"id1": id1, "id2": id2, "player": current_player})

                if new_squares_indices:
                    if current_player == 1:
                        list_squares_1.extend(new_squares_indices)
                    else:
                        list_squares_2.extend(new_squares_indices)
                else:
                    current_player = 3 - current_player

        SURF.fill(WHITE)
        score_p1_text = f"{player1_name}: {len(list_squares_1)}"
        score_p2_text = f"{player2_name}: {len(list_squares_2)}"
        score_p1_surf = score_font.render(score_p1_text, True, player1_color)
        score_p2_surf = score_font.render(score_p2_text, True, player2_color)
        SURF.blit(score_p1_surf, (start_x_p1, text_y))
        SURF.blit(score_p2_surf, (start_x_p2, text_y))

        for idx in list_squares_1:
            draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_BLUE)
        for idx in list_squares_2:
            draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_RED)
        
        for line_info in lines_drawn:
            draw_color = BLUE if line_info["player"] == 1 else RED
            draw_line(SURF, dot_positions, line_info["id1"], line_info["id2"], draw_color)

        for i, pos in enumerate(dot_positions):
            x, y = pos
            radius = DOT_RADIUS
            dot_color = BLACK
            if i == selected_dot_id:
                radius = DOT_HIGHLIGHT_RADIUS
                dot_color = GREEN
            gfxdraw.filled_circle(SURF, x, y, radius, dot_color)
            gfxdraw.aacircle(SURF, x, y, radius, dot_color)

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

def start_display(board_size_str, mode):
    try:
        parts = board_size_str.split('x')
        if len(parts) != 2:
            raise ValueError("Board size string must be in 'RowsxCols' format (e.g., '4x4')")
        rows = int(parts[0])
        cols = int(parts[1])
        if rows < 2 or cols < 2:
            print("Error: Board dimensions must be at least 2x2 to form squares.")
            return
        display_dots(rows, cols, mode)
    except (ValueError, IndexError) as e:
        print(f"Error parsing board size string '{board_size_str}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
