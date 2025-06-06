import pygame
import sys
from pygame import gfxdraw
import os
from collections import defaultdict, OrderedDict
import random
import math
import time
import numpy as np  # Add this import at the top

# Colors and game settings (unchanged)
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

# Zobrist hashing constants
MAX_EDGES = 100  # Maximum number of possible edges in a game
ZOBRIST_TABLE = None

# Add these constants
EXACT = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

# Optimal board-size specific evaluation values according to genetic algorithm
EVALUATION_VALUES = {
    '3x3': {
        'COMPLETE_SQUARE': 17.8,  
        'THREE_EDGE': 9.1,      
        'TWO_EDGE': 5.6,         
        'ONE_EDGE': 1.4,         
        'CHAIN_PENALTY': 9.8,    
        'CHAIN_DISRUPTION': 7.9,  
        'DOUBLE_CROSS': 15.6,    
        'CHAIN_LENGTH': 2.2,      
        'CHAIN_CREATION': 8.0,
        'CHAIN_LENGTH_THRESHOLD': 2,  # For 3x3, chains of 2+ are dangerous
        'SERIOUS_CHAIN_PENALTY': 150  # Less severe penalty for 3x3
    },
    '4x4': {
        'COMPLETE_SQUARE': 16.1, 
        'THREE_EDGE': 9.0,      
        'TWO_EDGE': 5.6,         
        'ONE_EDGE': 1.4,         
        'CHAIN_PENALTY': 11.5,    
        'CHAIN_DISRUPTION': 7.2,  
        'DOUBLE_CROSS': 12.1,     
        'CHAIN_LENGTH': 1.9,      
        'CHAIN_CREATION': 7.0,
        'CHAIN_LENGTH_THRESHOLD': 2,  # For 4x4, chains of 2+ are dangerous
        'SERIOUS_CHAIN_PENALTY': 200  # Standard penalty for 4x4
    },
    '4x5': {
        'COMPLETE_SQUARE': 18.0,  
        'THREE_EDGE': 6.2,     
        'TWO_EDGE': 3.5,        
        'ONE_EDGE': 1.4,        
        'CHAIN_PENALTY': 14.6,    
        'CHAIN_DISRUPTION': 3.6,  
        'DOUBLE_CROSS': 9.1,     
        'CHAIN_LENGTH': 2.3,      
        'CHAIN_CREATION': 4.9,
        'CHAIN_LENGTH_THRESHOLD': 3,  # For 4x5, chains of 3+ are dangerous
        'SERIOUS_CHAIN_PENALTY': 250  # Higher penalty for 4x5
    },
    '5x5': {
        'COMPLETE_SQUARE': 13.0,  
        'THREE_EDGE': 4.8,       
        'TWO_EDGE': 3.4,         
        'ONE_EDGE': 1.2,         
        'CHAIN_PENALTY': 16.5,    
        'CHAIN_DISRUPTION': 6.7,  
        'DOUBLE_CROSS': 10.5,     
        'CHAIN_LENGTH': 2.0,      
        'CHAIN_CREATION': 6.2,
        'CHAIN_LENGTH_THRESHOLD': 3,  # For 5x5, chains of 3+ are dangerous
        'SERIOUS_CHAIN_PENALTY': 300  # Highest penalty for 5x5
    }
}

def initialize_zobrist_table():
    """Initialize the Zobrist table with random 64-bit integers"""
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE is None:
        ZOBRIST_TABLE = np.random.randint(0, 2**64, size=(MAX_EDGES, 2), dtype=np.uint64)

def get_zobrist_hash(lines_drawn):
    """Calculate Zobrist hash for the current position"""
    if ZOBRIST_TABLE is None:
        initialize_zobrist_table()
    
    hash_value = np.uint64(0)
    for line in lines_drawn:
        # Create a unique index for each edge
        edge = tuple(sorted((line['id1'], line['id2'])))
        edge_index = edge[0] * 10 + edge[1]  # Simple hash function for edge index
        if edge_index >= MAX_EDGES:
            continue
        
        # XOR the hash with the corresponding Zobrist value
        hash_value ^= ZOBRIST_TABLE[edge_index][line['player'] - 1]
    
    return hash_value

# Replace the simple transposition table with a size-limited version using Zobrist hashing
class TranspositionTable:
    def __init__(self, max_size=500000):  # Restored to original size
        self.max_size = max_size
        self.table = OrderedDict()
        initialize_zobrist_table()
    
    def get(self, lines_drawn, depth, alpha, beta):
        key = get_zobrist_hash(lines_drawn)
        if key in self.table:
            entry = self.table[key]
            # Move to end (most recently used)
            self.table.pop(key)
            self.table[key] = entry
            
            # Only use the entry if it was searched at sufficient depth
            if entry['depth'] >= depth:
                if entry['type'] == EXACT:
                    return entry['score']
                if entry['type'] == LOWER_BOUND:
                    alpha = max(alpha, entry['score'])
                if entry['type'] == UPPER_BOUND:
                    beta = min(beta, entry['score'])
                if alpha >= beta:
                    return entry['score']
        return None
    
    def put(self, lines_drawn, score, depth, type, best_move=None):
        key = get_zobrist_hash(lines_drawn)
        entry = {
            'score': score,
            'depth': depth,
            'type': type,
            'best_move': best_move
        }
        
        if key in self.table:
            # Remove old entry
            self.table.pop(key)
        elif len(self.table) >= self.max_size:
            # Remove least recently used entry
            self.table.popitem(last=False)
        self.table[key] = entry
    
    def clear(self):
        self.table.clear()

transposition_table = TranspositionTable()  

# Unchanged utility functions
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

def is_square_controlled(square, edges):
    square_edges = edges_of_square(square)
    return all(edge in edges for edge in square_edges)

def get_opening_move(board_size, lines_drawn):
    # Enhanced opening strategy with more variations and responses
    opening_strategy = {
        '3x3': {
            'first_move': [(0,1), (0,3), (1,4), (3,6), (4,7)],  # Added more good first moves
            'responses': {
                (0,1): [(1,4), (0,3), (3,6)],  # Multiple good responses
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
        # First move - choose from multiple good options
        return random.choice(opening_strategy[key]['first_move'])
    elif len(lines_drawn) == 1:
        # Second move - respond to opponent's move
        last_move = tuple(sorted((lines_drawn[-1]['id1'], lines_drawn[-1]['id2'])))
        if last_move in opening_strategy[key]['responses']:
            return random.choice(opening_strategy[key]['responses'][last_move])
    
    return None

def get_evaluation_values(board_size):
    """Get evaluation values for the current board size"""
    key = f"{board_size[0]}x{board_size[1]}"
    if key not in EVALUATION_VALUES:
        # Default to 4x4 values if board size not found
        key = '4x4'
    return EVALUATION_VALUES[key]

def evaluate_position(lines_drawn, squares):
    # Check transposition table first
    cached_score = transposition_table.get(lines_drawn, 0, float('-inf'), float('inf'))
    if cached_score is not None:
        return cached_score
    
    score = 0
    current_edges = extract_edge_set(lines_drawn)
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0

    # Get board-size specific evaluation values
    board_size = get_board_size(squares)
    values = get_evaluation_values(board_size)

    # Evaluate squares based on edges
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            score += values['THREE_EDGE'] * (1 + game_phase * 0.5)
        elif len(existing) == 2:
            score += values['TWO_EDGE'] * (1 - game_phase * 0.3)
        elif len(existing) == 1:
            score += values['ONE_EDGE']

    # Chain evaluation
    three_edge_count = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
            # Check for chain patterns
            missing_edge = [edge for edge in edges if edge not in current_edges][0]
            temp_lines = lines_drawn.copy()
            temp_lines.append({"id1": missing_edge[0], "id2": missing_edge[1], "player": 2})
            if len(check_new_squares(missing_edge, temp_lines, squares)) > 0:
                score += values['CHAIN_CREATION'] * (1 - game_phase * 0.5)

    # Penalty for opponent-accessible three-edge squares
    opponent_three_edge = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            missing_edge = [edge for edge in edges if edge not in current_edges][0]
            temp_lines = lines_drawn + [{"id1": missing_edge[0], "id2": missing_edge[1], "player": 1}]
            if check_new_squares(missing_edge, temp_lines, squares):
                opponent_three_edge += 1
                score -= opponent_three_edge * values['CHAIN_PENALTY'] * (1 + game_phase)

    # Store in transposition table
    transposition_table.put(lines_drawn, score, 0, EXACT)
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

def predict_chain_reaction(move, lines_drawn, squares):
    bot_boxes = 0
    opponent_boxes = 0
    temp_lines = lines_drawn.copy()
    temp_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    bot_boxes += len(check_new_squares(move, lines_drawn, squares))
    current_edges = extract_edge_set(temp_lines)

    # Get board-size specific evaluation values
    board_size = get_board_size(squares)
    values = get_evaluation_values(board_size)

    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0

    three_edge_count = 0
    opponent_three_edge_count = 0
    favorable_chain_count = 0
    chain_length = 0
    max_chain_depth = 5
    
    # Track chain structure
    chain_starts = set()
    chain_ends = set()
    
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
            # Check if this would create a three-edge square for opponent
            missing_edge = [e for e in edges if e not in current_edges][0]
            temp_lines_opponent = temp_lines.copy()
            temp_lines_opponent.append({"id1": missing_edge[0], "id2": missing_edge[1], "player": 1})
            if len(check_new_squares(missing_edge, temp_lines_opponent, squares)) > 0:
                opponent_three_edge_count += 1
                chain_starts.add(missing_edge)
            else:
                # This is a favorable chain position
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

    # Calculate chain structure score
    chain_structure_score = 0
    if chain_starts and chain_ends:
        # If we have both starts and ends, we can control the chain
        chain_structure_score = len(chain_ends) * values['CHAIN_CREATION'] * (1 - game_phase * 0.5)
    elif chain_starts:
        # If we only have starts, we're giving away chains
        chain_structure_score = -len(chain_starts) * values['CHAIN_PENALTY'] * (1 + game_phase)

    # Scale chain penalty with chain length and opponent three-edge squares
    chain_structure_penalty = (three_edge_count * (1 + game_phase * 0.5) * values['CHAIN_PENALTY'] * 
                             (1 + chain_length * values['CHAIN_LENGTH']))
    opponent_chain_penalty = opponent_three_edge_count * values['CHAIN_PENALTY'] * (1 + game_phase)
    
    # Add bonus for favorable chain positions
    favorable_chain_bonus = favorable_chain_count * values['CHAIN_CREATION'] * (1 - game_phase * 0.5)
    
    disruption_bonus = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 2:
            disruption_bonus += values['CHAIN_DISRUPTION'] * (1 - game_phase * 0.3)

    return (bot_boxes - opponent_boxes - chain_structure_penalty - opponent_chain_penalty + 
            disruption_bonus + favorable_chain_bonus + chain_structure_score)

def minimax(lines_drawn, squares, depth, alpha, beta, is_maximizing):
    # Check transposition table
    cached_score = transposition_table.get(lines_drawn, depth, alpha, beta)
    if cached_score is not None:
        return cached_score
        
    if depth == 0 or is_game_over(lines_drawn, squares):
        score = evaluate_position(lines_drawn, squares)
        transposition_table.put(lines_drawn, score, depth, EXACT)
        return score

    if is_maximizing:
        max_eval = float('-inf')
        moves = get_valid_moves(lines_drawn, squares)
        moves.sort(key=lambda m: move_priority(m, lines_drawn, squares), reverse=True)
        best_move = None
        
        for move in moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            eval = minimax(new_lines, squares, depth-1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        
        # Store the result in transposition table
        if max_eval <= alpha:
            type = UPPER_BOUND
        elif max_eval >= beta:
            type = LOWER_BOUND
        else:
            type = EXACT
        transposition_table.put(lines_drawn, max_eval, depth, type, best_move)
        return max_eval
    else:
        min_eval = float('inf')
        moves = get_valid_moves(lines_drawn, squares)
        moves.sort(key=lambda m: move_priority(m, lines_drawn, squares), reverse=True)
        best_move = None
        
        for move in moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 1})
            eval = minimax(new_lines, squares, depth-1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        
        # Store the result in transposition table
        if min_eval <= alpha:
            type = UPPER_BOUND
        elif min_eval >= beta:
            type = LOWER_BOUND
        else:
            type = EXACT
        transposition_table.put(lines_drawn, min_eval, depth, type, best_move)
        return min_eval

def move_priority(move, lines_drawn, squares):
    new_lines = lines_drawn.copy()
    new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    
    # Get board-size specific evaluation values
    board_size = get_board_size(squares)
    values = get_evaluation_values(board_size)
    
    score = len(check_new_squares(move, lines_drawn, squares)) * values['COMPLETE_SQUARE']
    current_edges = extract_edge_set(new_lines)
    
    # Prioritize three-edge squares
    three_edge_count = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 3:
            three_edge_count += 1
    score += three_edge_count * values['THREE_EDGE']
    
    # Bonus for disrupting opponent two-edge squares
    disruption_score = 0
    for square in squares:
        edges = edges_of_square(square)
        existing = [e for e in edges if e in current_edges]
        if len(existing) == 2 and move in edges:
            disruption_score += values['CHAIN_DISRUPTION']
    score += disruption_score
    
    return score

def bot_choose_move(lines_drawn, squares):
    # Calculate game phase
    current_edges = extract_edge_set(lines_drawn)
    total_squares = len(squares)
    completed_squares = sum(1 for square in squares if is_square_controlled(square, current_edges))
    game_phase = completed_squares / total_squares if total_squares > 0 else 0
    
    # Get board-size specific evaluation values
    board_size = get_board_size(squares)
    values = get_evaluation_values(board_size)
    
    # Count three-edge squares
    three_edge_count = sum(1 for square in squares 
                          if len([e for e in edges_of_square(square) if e in current_edges]) == 3)

    # Check opening move
    opening_move = get_opening_move(get_board_size(squares), lines_drawn)
    if opening_move:
        return opening_move

    # Get valid moves and sort by priority
    valid_moves = get_valid_moves(lines_drawn, squares)
    if not valid_moves:
        return None

    board_size = get_board_size(squares)
    remaining_moves = len(valid_moves)
    
    # Adjust max depth based on game state - using grok's logic
    base_depth = max(3, int(14 - remaining_moves / (board_size[0] * board_size[1] * 0.1)))
    if game_phase > 0.7 or three_edge_count >= 2:  # Critical position or endgame
        max_depth = min(10, base_depth + 3)  # Look deeper in critical positions
    else:
        max_depth = min(6, base_depth)

    valid_moves.sort(key=lambda m: move_priority(m, lines_drawn, squares), reverse=True)

    best_move = None
    best_score = float('-inf')
    best_moves = []
    score_threshold = 0.6

    depth = 2  # Start with a shallow search like grok
    while depth <= max_depth:
        alpha = float('-inf')
        beta = float('inf')
        temp_best_score = float('-inf')
        temp_best_moves = []

        for move in valid_moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            
            # Check if this move completes a box
            completed_boxes = len(check_new_squares(move, lines_drawn, squares))
            
            # If this move completes a box, check for chain creation
            if completed_boxes > 0:
                # Check if this creates a three-edge square for opponent
                three_edge_for_opponent = False
                opponent_chain_length = 0
                for square in squares:
                    edges = edges_of_square(square)
                    existing = [e for e in edges if e in extract_edge_set(new_lines)]
                    if len(existing) == 3:
                        missing_edge = [e for e in edges if e not in extract_edge_set(new_lines)][0]
                        if missing_edge not in extract_edge_set(lines_drawn):
                            three_edge_for_opponent = True
                            # Calculate potential chain length
                            temp_lines = new_lines.copy()
                            temp_lines.append({"id1": missing_edge[0], "id2": missing_edge[1], "player": 1})
                            opponent_chain_length = predict_chain_reaction(missing_edge, temp_lines, squares)
                
                if three_edge_for_opponent and opponent_chain_length > values['CHAIN_LENGTH_THRESHOLD']:
                    eval = -values['SERIOUS_CHAIN_PENALTY'] * opponent_chain_length
                else:
                    eval = minimax(new_lines, squares, depth, alpha, beta, False)
            else:
                eval = minimax(new_lines, squares, depth, alpha, beta, False)
            
            chain_score = predict_chain_reaction(move, lines_drawn, squares)
            total_score = eval + chain_score

            if total_score > temp_best_score + score_threshold:
                temp_best_score = total_score
                temp_best_moves = [move]
            elif abs(total_score - temp_best_score) <= score_threshold:
                temp_best_moves.append(move)
            alpha = max(alpha, total_score)

        if temp_best_moves:
            best_score = temp_best_score
            best_moves = temp_best_moves
            best_move = random.choice(best_moves)

        depth += 1

    return best_move if best_move else random.choice(valid_moves) if valid_moves else None


# Remaining functions (unchanged)
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
    if not (0 <= id1 < num_dots and 0 <= id2 < num_dots): return
    try:
        vt1 = dot_positions[id1]
        vt2 = dot_positions[id2]
        pygame.draw.line(surface, color, vt1, vt2, thickness)
    except IndexError:
        print(f"Error drawing line: Index out of range for IDs ({id1}, {id2})")

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
        print("Minimum display size might be too small for the requested grid.")
        pygame.quit()
        return 
    
    pygame.display.set_caption("Dots and Boxes")
    
    icon_path = os.path.join(os.path.dirname(__file__), 'images', 'dotsandboxes.png')
    try:
        if os.path.exists(icon_path):
             icon_surf = pygame.image.load(icon_path)
             pygame.display.set_icon(icon_surf)
             print(f"Icon loaded successfully from: {icon_path}")
        else:
             print(f"Warning: Icon file not found at {icon_path}")
    except pygame.error as e:
        print(f"Warning: Could not load or set icon from {icon_path}. Error: {e}")
    except Exception as e:
        print(f"Warning: An unexpected error occurred while handling the icon: {e}")

    try:
        score_font = pygame.font.SysFont('Arial', 30)
    except:
        print("Warning: Arial font not found, using default.")
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

            if event.type == pygame.MOUSEBUTTONDOWN and current_player == 1:
                mouse_x, mouse_y = event.pos
                clicked_dot_id = get_point_at_vt(dot_positions, mouse_x, mouse_y)

                if clicked_dot_id is not None:
                    if selected_dot_id is None:
                        selected_dot_id = clicked_dot_id
                    elif selected_dot_id == clicked_dot_id:
                        print("Clicked same dot, deselecting.") 
                        selected_dot_id = None
                    else:
                        id1 = selected_dot_id
                        id2 = clicked_dot_id

                        is_adjacent = False
                        vt1 = dot_positions[id1]
                        vt2 = dot_positions[id2]

                        if abs(vt1[1] - vt2[1]) < 5 and abs(vt1[0] - vt2[0] - GRID_SPACING) < 5 or \
                           abs(vt1[1] - vt2[1]) < 5 and abs(vt1[0] - vt2[0] + GRID_SPACING) < 5 :
                           is_adjacent = True

                        elif abs(vt1[0] - vt2[0]) < 5 and abs(vt1[1] - vt2[1] - GRID_SPACING) < 5 or \
                             abs(vt1[0] - vt2[0]) < 5 and abs(vt1[1] - vt2[1] + GRID_SPACING) < 5:
                             is_adjacent = True

                        if is_adjacent:
                            id_pair = tuple(sorted((id1, id2)))
                            line_exists = any(line['id1'] == id_pair[0] and line['id2'] == id_pair[1] for line in lines_drawn)

                            if not line_exists:
                                new_squares_indices = check_new_squares(id_pair, lines_drawn, squares)
                                lines_drawn.append({"id1": id_pair[0], "id2": id_pair[1], "player": current_player})
                                print(lines_drawn)

                                if new_squares_indices:
                                    list_squares_1.extend(new_squares_indices)
                                else:
                                    current_player = 2  # Switch to AI
                                    
                                # Update display immediately after player's move
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
                                    if line_info["player"] == 1:
                                        draw_color = BLUE
                                    else:
                                        draw_color = RED
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
                            else:
                                print("Line already exists.") 
                        else:
                            print("Dots are not adjacent, line not added.") 
                        selected_dot_id = None
                else: 
                    print("Clicked empty space, deselecting.")
                    selected_dot_id = None

        # Bot turn
        if current_player == 2:
            move = bot_choose_move(lines_drawn, squares)
            if move:
                id1, id2 = move
                new_squares_indices = check_new_squares((id1, id2), lines_drawn, squares)
                lines_drawn.append({"id1": id1, "id2": id2, "player": 2})
                print(lines_drawn)
                print(f"Bot added line: {id1}-{id2}")

                if new_squares_indices:
                    list_squares_2.extend(new_squares_indices)
                else:
                    current_player = 1  # Switch back to player

        # Update display for both player and AI moves
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
            if line_info["player"] == 1:
                draw_color = BLUE
            else:
                draw_color = RED
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
            raise ValueError("Board size string must be in 'RowsxCols' format (e.g., '4x5')")
        rows = int(parts[0])
        cols = int(parts[1])
        if rows < 1 or cols < 1:
            print("Error: Board dimensions must be at least 1x1.")
            return
        display_dots(rows, cols, mode)
    except (ValueError, IndexError) as e:
        print(f"Error parsing board size string '{board_size_str}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")