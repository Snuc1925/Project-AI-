import math
import pygame
import sys
from pygame import gfxdraw 
import os
from collections import defaultdict
import random
import time
import zlib

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
LINE_COLOR
DOT_HIGHLIGHT_RADIUS = 7

# Constants for transposition table
EXACT = 0  # Exact score
LOWERBOUND = 1  # Lower bound score
UPPERBOUND = 2  # Upper bound score

class TranspositionTable:
    def __init__(self, max_size=1000000):  # Limit to 1 million entries
        self.table = {}
        self.max_size = max_size
        self.access_count = {}  # Track how often each entry is accessed
        
    def get(self, key):
        if key in self.table:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.table[key]
        return None
        
    def put(self, key, value, flag, depth):
        # If table is full, remove least accessed entries
        if len(self.table) >= self.max_size:
            # Remove 10% of least accessed entries
            num_to_remove = self.max_size // 10
            sorted_entries = sorted(self.access_count.items(), key=lambda x: x[1])
            for k, _ in sorted_entries[:num_to_remove]:
                del self.table[k]
                del self.access_count[k]
        
        self.table[key] = (value, flag, depth)
        self.access_count[key] = 0
        
    def clear(self):
        self.table.clear()
        self.access_count.clear()

class EvaluationParameters:
    def __init__(self, board_size):
        # Default parameters for each board size
        self.params = {
            "3x3": {
                "multiple_squares_bonus": 2125.0988841700746,
                "chain_penalty": -1907.9751592795346,
                "double_cross_bonus": 1728.2470762223834,
                "safe_move_bonus": 1990.5025495556354,
                "center_bonus": 543.4994758905656,
                "chain_threshold": 6,
                "single_square_bonus": 1410.6350535953727
            },
            "2x2": {
                "multiple_squares_bonus": 1000,
                "chain_penalty": -500,
                "double_cross_bonus": 800,
                "safe_move_bonus": 300,
                "center_bonus": 100,
                "chain_threshold": 1,
                "single_square_bonus": 100
            },
            "3x4": {
                "multiple_squares_bonus": 1000,
                "chain_penalty": -500,
                "double_cross_bonus": 800,
                "safe_move_bonus": 300,
                "center_bonus": 100,
                "chain_threshold": 1,
                "single_square_bonus": 100
            },
            "4x4": {
                "multiple_squares_bonus": 1122.0833881389435,
                "chain_penalty": -459.42278109698145,
                "double_cross_bonus": 2905.4619406456823,
                "safe_move_bonus": 774.0564282613578,
                "center_bonus": 134.3534846179324,
                "chain_threshold": 5,
                "single_square_bonus": 1110.7746685027942
            }
        }
        self.board_size = board_size

    def get_params(self):
        return self.params[self.board_size]

    def update_params(self, new_params):
        self.params[self.board_size] = new_params

# Global parameter instance
eval_params = EvaluationParameters("3x3")

def get_point_at_vt(dot_positions, x, y, click_radius=CLICK_RADIUS):
    for i, vt in enumerate(dot_positions):
        px, py = vt
        dist = (px - x)**2 + (py - y)**2
        if dist < click_radius**2:
            return i
    return None 

def draw_line(surface, dot_positions, id1, id2, color, thickness=3):
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

def edges_of_square(square):
    a, b, c, d = square
    return [tuple(sorted(edge)) for edge in [(a, b), (a, c), (b, d), (c, d)]]

def extract_edge_set(lines_drawn):
    return set(tuple(sorted((line['id1'], line['id2']))) for line in lines_drawn)

def draw_colored_squares(idx, squares, dot_positions, canvas, color=BLACK):
    a, b, c, d = squares[idx]
    points = [dot_positions[a], dot_positions[b], dot_positions[d], dot_positions[c]]  # theo chiều kim đồng hồ
    pygame.draw.polygon(canvas, color, points)

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

def get_key(lines_drawn):
    key = 0
    for line in lines_drawn:
        key ^= (line['id1'] * 31 + line['id2']) * 17
    return key

def evaluate_state_alphabeta(lines_drawn, squares, list_squares_1, list_squares_2, is_maximizing):
    # Basic score = bot score minus player score
    score = len(list_squares_2) - len(list_squares_1)
    
    # Add chain bonus
    score += evaluate_chain(lines_drawn, squares, is_maximizing)
    
    # Add potential bonus (squares close to completion)
    score += evaluate_potential(lines_drawn, squares, is_maximizing)
    
    # Add safety score (avoid giving opponent easy squares)
    score += evaluate_safety(lines_drawn, squares)
    
    return score

def evaluate_chain(lines_drawn, squares, is_maximizing):
    chain_bonus = 0
    current_edges = extract_edge_set(lines_drawn)
    
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            if is_part_of_chain(lines_drawn, squares, square):
                chain_bonus += 1 if is_maximizing else -1
    return chain_bonus

def evaluate_potential(lines_drawn, squares, is_maximizing):
    """
    Tính điểm tiềm năng dựa trên số cạnh đã vẽ:
      - +2.0 nếu ô có 3 cạnh đã vẽ
      - +0.5 nếu ô có 2 cạnh đã vẽ
      -  0.0 cho trường hợp khác

    Dương cho lượt bot, âm cho lượt người chơi.
    """
    potential_bonus = 0.0
    current_edges = extract_edge_set(lines_drawn)
    
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            drawn = sum(1 for edge in edges if edge in current_edges)
            if drawn == 3:
                bonus = 2.0
            elif drawn == 2:
                bonus = 0.5
            else:
                bonus = 0
            potential_bonus += bonus if is_maximizing else -bonus
    return potential_bonus

def evaluate_safety(lines_drawn, squares):
    safety_score = 0
    current_edges = extract_edge_set(lines_drawn)
    
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            if is_safe_move(lines_drawn, squares, square):
                safety_score += 1  # Safe moves are valued higher
            else:
                safety_score -= 1  # If it's a dangerous move, subtract points
    return safety_score

def is_part_of_chain(lines_drawn, squares, square):

    current_edges = extract_edge_set(lines_drawn)
    square_edges = edges_of_square(square)
    
    # Get adjacent squares
    for adj_square in squares:
        if adj_square != square:
            adj_edges = edges_of_square(adj_square)
            if not all(edge in current_edges for edge in adj_edges):
                drawn = sum(1 for edge in adj_edges if edge in current_edges)
                if drawn == 3:
                    return True
    return False

def is_safe_move(lines_drawn, squares, move):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(move)  # Add the move to check its effect
    
    # Check adjacent squares
    three_edge_count = 0
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            drawn = sum(1 for edge in edges if edge in current_edges)
            if drawn == 3:
                three_edge_count += 1
                if three_edge_count > 1:  # More than one three-edge square is dangerous
                    return False
    
    # If we have exactly one three-edge square, it might be a double-cross opportunity
    if three_edge_count == 1:
        return True
        
    return three_edge_count == 0  # Safe if no three-edge squares are created

def alphabeta(lines_drawn, squares, list_squares_1, list_squares_2, depth, alpha, beta, is_maximizing, transposition_table=None):
    """
    alphabeta algorithm with alpha-beta pruning and transposition table.

    Parameters:
      lines_drawn      – list of drawn lines
      squares          – list of all possible squares
      list_squares_1   – list of squares completed by player 1
      list_squares_2   – list of squares completed by player 2
      depth            – remaining depth
      alpha, beta      – pruning thresholds
      is_maximizing    – True if bot's turn, False if player's turn
      transposition_table – table to store evaluated positions

    Returns:
      (best_score, best_move) – tuple of best score and corresponding move
    """
    # Only use transposition table for higher depths to improve cache hit rate
    if transposition_table and depth >= 2:
        key = get_key(lines_drawn)
        entry = transposition_table.get(key)
        if entry:
            stored_value, flag, stored_depth = entry
            if stored_depth >= depth:
                if flag == EXACT:
                    return stored_value, None
                elif flag == LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif flag == UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, None

    # Stop at depth 0 or game over
    if depth == 0 or is_game_over(lines_drawn, squares):
        eval_score = evaluate_state_alphabeta(lines_drawn, squares, list_squares_1, list_squares_2, is_maximizing)
        if transposition_table and depth >= 2:
            transposition_table.put(key, eval_score, EXACT, depth)
        return eval_score, None

    moves = get_valid_moves(lines_drawn, squares)
    if not moves:
        return 0, None

    # Sort moves by potential value (heuristic)
    moves.sort(key=lambda m: evaluate_move(m, lines_drawn, squares, is_maximizing), reverse=is_maximizing)
    
    best_move = None
    original_alpha = alpha

    if is_maximizing:
        max_eval = -math.inf
        for move in moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            
            # Check if move completes any squares
            new_squares = check_new_squares(move, lines_drawn, squares)
            new_list_squares_2 = list_squares_2.copy()
            new_list_squares_2.extend(new_squares)
            
            if new_squares:
                # Bot gets another turn
                val, _ = alphabeta(new_lines, squares, list_squares_1, new_list_squares_2, depth, alpha, beta, True, transposition_table)
            else:
                # Switch to player's turn
                val, _ = alphabeta(new_lines, squares, list_squares_1, list_squares_2, depth-1, alpha, beta, False, transposition_table)
                
            if val > max_eval:
                max_eval, best_move = val, move
            alpha = max(alpha, val)
            if beta <= alpha:
                break  # pruning
                
        if transposition_table and depth >= 2:
            flag = EXACT
            if max_eval <= original_alpha:
                flag = UPPERBOUND
            elif max_eval >= beta:
                flag = LOWERBOUND
            transposition_table.put(key, max_eval, flag, depth)
            
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 1})
            
            # Check if move completes any squares
            new_squares = check_new_squares(move, lines_drawn, squares)
            new_list_squares_1 = list_squares_1.copy()
            new_list_squares_1.extend(new_squares)
            
            if new_squares:
                val, _ = alphabeta(new_lines, squares, new_list_squares_1, list_squares_2, depth, alpha, beta, False, transposition_table)
            else:
                val, _ = alphabeta(new_lines, squares, list_squares_1, list_squares_2, depth-1, alpha, beta, True, transposition_table)
                
            if val < min_eval:
                min_eval, best_move = val, move
            beta = min(beta, val)
            if beta <= alpha:
                break  # pruning
                
        if transposition_table and depth >= 2:
            flag = EXACT
            if min_eval <= original_alpha:
                flag = UPPERBOUND
            elif min_eval >= beta:
                flag = LOWERBOUND
            transposition_table.put(key, min_eval, flag, depth)
            
        return min_eval, best_move

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
    """
    Checks if the game is over by seeing if all squares are completed.
    """
    current_edges = extract_edge_set(lines_drawn)
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            return False
    return True

def evaluate_move(move, lines_drawn, squares, is_maximizing):
    """
    Prioritize moves based on proven Dots and Boxes strategies using optimized parameters.
    """
    params = eval_params.get_params()
    
    # Simulate the move
    new_lines = lines_drawn.copy()
    new_lines.append({"id1": move[0], "id2": move[1], "player": 2 if is_maximizing else 1})
    current_edges = extract_edge_set(new_lines)
    
    # Check if move completes any squares
    new_squares = check_new_squares(move, new_lines, squares)
    if new_squares:
        # If it completes multiple squares, evaluate chain potential
        if len(new_squares) > 1:
            # Check if this creates a chain for opponent
            chain_length = 0
            for square in squares:
                edges = edges_of_square(square)
                if not all(edge in current_edges for edge in edges):
                    drawn = sum(1 for edge in edges if edge in current_edges)
                    if drawn == 3:
                        chain_length += 1
            
            # If creates a long chain for opponent, lower priority
            if chain_length > params["chain_threshold"]:
                return params["chain_penalty"]
            return params["multiple_squares_bonus"]
        # Single square completion is third priority
        return params["single_square_bonus"]

    # Count three-edge squares for opponent
    opponent_three_edges = 0
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges):
            drawn = sum(1 for edge in edges if edge in current_edges)
            if drawn == 3:
                opponent_three_edges += 1
    
    # Check for double-cross opportunities
    if opponent_three_edges == 1:
        return params["double_cross_bonus"]
    
    # Check if move is safe
    if is_safe_move(new_lines, squares, move):
        return params["safe_move_bonus"]

    # Check for center control
    center_bonus = 0
    rows = int(math.sqrt(len(squares))) + 1
    cols = int(math.sqrt(len(squares))) + 1
    center_row = rows // 2
    center_col = cols // 2
    
    # Check if move is in or near center
    move_row1 = move[0] // cols
    move_col1 = move[0] % cols
    move_row2 = move[1] // cols
    move_col2 = move[1] % cols
    
    if (abs(move_row1 - center_row) <= 1 and abs(move_col1 - center_col) <= 1) or \
       (abs(move_row2 - center_row) <= 1 and abs(move_col2 - center_col) <= 1):
        center_bonus = params["center_bonus"]

    return center_bonus

def bot_choose_move(lines_drawn, squares, list_squares_1=None, list_squares_2=None):
    """
    Bot's move selection using iterative deepening alphabeta algorithm.
    """
    if list_squares_1 is None:
        list_squares_1 = []
    if list_squares_2 is None:
        list_squares_2 = []
        
    # Get valid moves
    moves = get_valid_moves(lines_drawn, squares)
    if not moves:
        return None
        
    # Initialize transposition table with smaller size
    transposition_table = TranspositionTable(max_size=500000)
    
    # Iterative deepening with time limit
    max_depth = 3
    best_move = None
    
    # First, evaluate all moves to find forced moves and chain opportunities
    move_scores = []
    for move in moves:
        score = evaluate_move(move, lines_drawn, squares, True)
        move_scores.append((move, score))
    
    # Sort moves by score, but don't automatically take the highest scoring move
    move_scores.sort(key=lambda x: x[1], reverse=True)
    moves = [m[0] for m in move_scores]
    
    for depth in range(1, max_depth + 1):
        alpha = -math.inf
        beta = math.inf
        
        current_best_score = -math.inf
        current_best_move = None
        
        for move in moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            
            # Check if move completes any squares
            new_squares = check_new_squares(move, lines_drawn, squares)
            new_list_squares_2 = list_squares_2.copy()
            new_list_squares_2.extend(new_squares)
            
            if new_squares:
                # Bot gets another turn
                score, _ = alphabeta(new_lines, squares, list_squares_1, new_list_squares_2, depth, alpha, beta, True, transposition_table)
            else:
                # Switch to player's turn
                score, _  = alphabeta(new_lines, squares, list_squares_1, list_squares_2, depth-1, alpha, beta, False, transposition_table)
                
            if score > current_best_score:
                current_best_score = score
                current_best_move = move
            alpha = max(alpha, score)
            
        # Update best move if we found a better one at this depth
        if current_best_move:
            best_move = current_best_move
            
    return best_move

def display_dots(rows, cols, mode, bot1, bot2, on_back_to_menu=None, bot1_name=None, bot2_name=None): 
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
        player2_name = bot1_name if bot1_name else "AI"
        player1_color = BLUE
        player2_color = RED
    elif mode == "AI vs AI":
        player1_name = bot1_name if bot1_name else "AI1"
        player2_name = bot2_name if bot2_name else "AI2"
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

    if mode == "AI vs AI":
        is_player1_human = False
    else:
        is_player1_human = True

    is_player2_human = False

    print(mode, is_player1_human, is_player2_human)

    # Add variables for move highlighting
    last_move_dots = None
    last_move_player = None  # Track which player made the last move

    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Xử lý input cho human player
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Kiểm tra xem có phải lượt của human không
                is_current_player_human = (current_player == 1 and is_player1_human) or (current_player == 2 and is_player2_human)
                
                if is_current_player_human:
                    mouse_x, mouse_y = event.pos
                    clicked_dot_id = get_point_at_vt(dot_positions, mouse_x, mouse_y)

                    if clicked_dot_id is not None:
                        if selected_dot_id is None:
                            # Chọn dot đầu tiên
                            selected_dot_id = clicked_dot_id
                        elif selected_dot_id == clicked_dot_id:
                            # Bỏ chọn dot
                            print("Clicked same dot, deselecting.") 
                            selected_dot_id = None
                        else:
                            # Thử vẽ line giữa 2 dots
                            id1 = selected_dot_id
                            id2 = clicked_dot_id

                            # Kiểm tra 2 dots có kề nhau không
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
                                    # Vẽ line và kiểm tra squares mới
                                    new_squares_indices = check_new_squares(id_pair, lines_drawn, squares)
                                    lines_drawn.append({"id1": id_pair[0], "id2": id_pair[1], "player": current_player})
                                    print(f"Player {current_player} added line: {id_pair}")

                                    # Cập nhật squares và chuyển lượt
                                    if new_squares_indices:
                                        if current_player == 1:
                                            list_squares_1.extend(new_squares_indices)
                                        else:
                                            list_squares_2.extend(new_squares_indices)
                                        # Không chuyển lượt nếu tạo được square
                                    else:
                                        # Chuyển lượt
                                        current_player = 2 if current_player == 1 else 1
                                else:
                                    print("Line already exists.") 
                            else:
                                print("Dots are not adjacent, line not added.") 
                            selected_dot_id = None
                    else: 
                        print("Clicked empty space, deselecting.")
                        selected_dot_id = None

        # Xử lý lượt AI
        is_current_player_ai = (current_player == 1 and not is_player1_human) or (current_player == 2 and not is_player2_human)
        
        if is_current_player_ai:
            # Add delay in AI vs AI mode
            if mode == "AI vs AI":
                pygame.time.wait(750)  # 750ms delay between AI moves
                
            # Chọn bot phù hợp
            if current_player == 1:
                current_bot = bot1
            else:
                current_bot = bot2 if mode == "AI vs AI" else bot1
            
            move = current_bot(lines_drawn, squares, list_squares_1, list_squares_2)

            if move:
                id1, id2 = move
                new_squares_indices = check_new_squares((id1, id2), lines_drawn, squares)
                lines_drawn.append({"id1": id1, "id2": id2, "player": current_player})
                print(f"AI {current_player} added line: {id1}-{id2}")

                # Set highlight for the last move
                last_move_dots = (id1, id2)
                last_move_player = current_player

                # Cập nhật squares và chuyển lượt
                if new_squares_indices:
                    if current_player == 1:
                        list_squares_1.extend(new_squares_indices)
                    else:
                        list_squares_2.extend(new_squares_indices)
                    # AI tiếp tục nếu tạo được square
                else:
                    # Chuyển lượt
                    current_player = 2 if current_player == 1 else 1

        # Vẽ màn hình
        SURF.fill(WHITE) 

        # Vẽ điểm số
        score_p1_text = f"{player1_name}: {len(list_squares_1)}"
        score_p2_text = f"{player2_name}: {len(list_squares_2)}"

        score_p1_surf = score_font.render(score_p1_text, True, player1_color)
        score_p2_surf = score_font.render(score_p2_text, True, player2_color)
        
        # Tính vị trí hiển thị điểm số
        p1_width, p1_height = score_p1_surf.get_size()
        p2_width, p2_height = score_p2_surf.get_size()
        spacing = 50 
        total_width = p1_width + spacing + p2_width
        start_x_p1 = (screen_width - total_width) // 2
        start_x_p2 = start_x_p1 + p1_width + spacing
        text_y = 30
        
        SURF.blit(score_p1_surf, (start_x_p1, text_y))
        SURF.blit(score_p2_surf, (start_x_p2, text_y))

        # Vẽ colored squares
        for idx in list_squares_1:
            draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_BLUE)
        for idx in list_squares_2:
            draw_colored_squares(idx, squares, dot_positions, SURF, LIGHT_RED)
        
        # Vẽ lines
        for line_info in lines_drawn:
            if line_info["player"] == 1:
                draw_color = player1_color
            else:
                draw_color = player2_color
            draw_line(SURF, dot_positions, line_info["id1"], line_info["id2"], draw_color)

        # Vẽ dots with highlight for last move
        for i, pos in enumerate(dot_positions):
            x, y = pos
            radius = DOT_RADIUS
            dot_color = BLACK

            if i == selected_dot_id:
                radius = DOT_HIGHLIGHT_RADIUS
                dot_color = GREEN
            elif last_move_dots and (i == last_move_dots[0] or i == last_move_dots[1]):
                radius = DOT_HIGHLIGHT_RADIUS
                # Use player color for the highlight
                dot_color = player1_color if last_move_player == 1 else player2_color

            gfxdraw.filled_circle(SURF, x, y, radius, dot_color)
            gfxdraw.aacircle(SURF, x, y, radius, dot_color)

        pygame.display.update()
        
        # Kiểm tra kết thúc game
        if len(list_squares_1) + len(list_squares_2) == len(squares):
            overlay = pygame.Surface((screen_width, screen_height))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))

            # Xác định kết quả
            if len(list_squares_1) > len(list_squares_2):
                result_text = f"{player1_name} wins!"
            elif len(list_squares_1) < len(list_squares_2):
                result_text = f"{player2_name} wins!"
            else:
                result_text = "It's a draw!"

            result_font = pygame.font.SysFont('Arial', 60)
            result_surf = result_font.render(result_text, True, WHITE) 
            result_rect = result_surf.get_rect(center=(screen_width // 2, screen_height // 2 - 50))
            
            SURF.blit(overlay, (0, 0)) 
            SURF.blit(result_surf, result_rect)

            button_font = pygame.font.SysFont("Arial", 25)
            button_text_content = "Go back to Menu"
            button_color = (0, 128, 255)
            button_hover_color = (50, 178, 255)
            button_width = 200
            button_height = 40
            button_rect = pygame.Rect(0, 0, button_width, button_height)
            button_rect.centerx = screen_width // 2
            button_rect.bottom = screen_height - 30 

            button_text_surf = button_font.render(button_text_content, True, WHITE)
            button_text_rect = button_text_surf.get_rect(center=button_rect.center)
            
            # Đổi màu khi hover vào nút
            mouse_pos = pygame.mouse.get_pos()
            current_button_color = button_hover_color if button_rect.collidepoint(mouse_pos) else button_color
            pygame.draw.rect(SURF, current_button_color, button_rect, border_radius=18)
            SURF.blit(button_text_surf, button_text_rect)
            pygame.display.update()

            # Chờ user action
            waiting_for_action = True
            while waiting_for_action:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_action = False
                        running = False 
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and button_rect.collidepoint(event.pos):
                            waiting_for_action = False
                            running = False 
                            if on_back_to_menu: on_back_to_menu()

        clock.tick(60)

    pygame.quit()
    

def start_display(board_size_str, mode, bot1=None, bot2=None, on_back=None, bot1_name=None, bot2_name=None):
    try:
        parts = board_size_str.split('x')
        if len(parts) != 2:
            raise ValueError("Board size string must be in 'RowsxCols' format (e.g., '4x5')")
        rows = int(parts[0])
        cols = int(parts[1])
        if rows < 1 or cols < 1:
            print("Error: Board dimensions must be at least 1x1.")
            return
            
        # Convert to box size for parameter loading
        box_size = f"{rows-1}x{cols-1}"
        
        # Try to load optimized parameters
        try:
            from parameter_optimizer import load_optimized_parameters
            params = load_optimized_parameters(box_size)
            if params:
                global eval_params
                eval_params = EvaluationParameters(box_size)
                eval_params.update_params(params)
                print(f"Loaded optimized parameters for {box_size} board")
        except Exception as e:
            print(f"Could not load optimized parameters: {e}")
            print("Using default parameters")
            
        display_dots(rows, cols, mode, bot1, bot2, on_back, bot1_name, bot2_name)
    except (ValueError, IndexError) as e:
        print(f"Error parsing board size string '{board_size_str}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


