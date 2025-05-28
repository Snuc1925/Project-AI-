# menelaus's implementation
import pygame
import sys
from pygame import gfxdraw 
import os
from collections import defaultdict
import random
import math

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)  
LIGHT_RED = (255, 114, 114)
BLUE = (0, 0, 255)  
LIGHT_BLUE = (148, 249, 237)
GREEN = (0, 200, 0)

# game settings
DOT_RADIUS = 5
GRID_SPACING = 100 
MARGIN = 100 
CLICK_RADIUS = 15
LINE_COLOR = BLACK
DOT_HIGHLIGHT_RADIUS = 7
THICKNESS = 3

# static evaluation values
COMPLETE_SQUARE_VALUE = 10
THREE_EDGE_VALUE = 6
TWO_EDGE_VALUE = 2
ONE_EDGE_VALUE = 1
CENTER_CONTROL_VALUE = 3.5
CHAIN_PENALTY = 4.5

# mouse click handling
def get_point_at_vt(dot_positions, x, y, click_radius = CLICK_RADIUS):
    for i, vt in enumerate(dot_positions):
        px, py = vt
        dist = (px - x)**2 + (py - y)**2
        if dist < click_radius**2:
            return i
    return None 

# lines drawing (between 2 dots)
def draw_line(surface, dot_positions, id1, id2, color, thickness = THICKNESS):
    num_dots = len(dot_positions)
    if not (0 <= id1 < num_dots and 0 <= id2 < num_dots): return
    try:
        vt1 = dot_positions[id1]
        vt2 = dot_positions[id2]
        pygame.draw.line(surface, color, vt1, vt2, thickness)
    except IndexError:
        print(f"Error drawing line: Index out of range for IDs ({id1}, {id2})")

# squares generation (between 4 dots)
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

# get all valid moves (for AI to choose)
def get_valid_moves(lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    valid_moves = set() # all valid moves
    
    for square in squares:
        edges = edges_of_square(square)
        for edge in edges:
            if edge not in current_edges:
                valid_moves.add(edge)
    
    return list(valid_moves)

# check if the game is over
def is_game_over(lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    for square in squares:
        edges = edges_of_square(square)
        if not all(edge in current_edges for edge in edges): # if one incompleted squared is founded...
            return False # ...the game is not over
    return True

# return size of the board
def get_board_size(squares):
    if not squares:
        return (0, 0)
    max_row = max(max(square) for square in squares) // (len(squares) + 1)
    max_col = max(max(square) for square in squares) % (len(squares) + 1)
    return (max_row + 1, max_col + 1)

# return list of center squares
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

def evaluate_position(lines_drawn, squares):
    score = 0
    current_edges = extract_edge_set(lines_drawn)
    
    # count potential squares
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in current_edges]
        if len(existing) == 3:
            score += COMPLETE_SQUARE_VALUE
        elif len(existing) == 2:
            score += TWO_EDGE_VALUE
        elif len(existing) == 1:
            score += ONE_EDGE_VALUE
    
    # consider center control
    center_squares = get_center_squares(squares)
    for square in center_squares:
        if is_square_controlled(square, current_edges):
            score += CENTER_CONTROL_VALUE
    
    return score

# predict how many squares can be completed if the move is made
def predict_chain_reaction(move, lines_drawn, squares):
    chain_length = 0
    temp_lines = lines_drawn.copy() # copy the current game state...
    temp_lines.append({"id1": move[0], "id2": move[1], "player": 2}) # ...and add the move
    
    while True:
        forced_moves = find_forced_moves(temp_lines, squares)
        if not forced_moves:
            break
        chain_length += len(forced_moves)
        for forced_move in forced_moves:
            temp_lines.append({"id1": forced_move[0], "id2": forced_move[1], "player": 2})
    
    return chain_length

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

def get_opening_move(board_size, lines_drawn):
    opening_moves = {
        '3x3': [(0,1), (1,4), (4,5), (3,6)],
        '4x4': [(0,1), (1,5), (5,6), (4,8)],
        '5x5': [(0,1), (1,6), (6,7), (5,11)],
        '4x5': [(0,1), (1,5), (5,6), (4,9)]
    }
    
    if len(lines_drawn) < 4:  # only use opening book in early game
        key = f"{board_size[0]}x{board_size[1]}"
        if key in opening_moves:
            return opening_moves[key][len(lines_drawn)]
    return None

def minimax(lines_drawn, squares, depth, alpha, beta, is_maximizing):
    if depth == 0 or is_game_over(lines_drawn, squares):
        return evaluate_position(lines_drawn, squares)
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in get_valid_moves(lines_drawn, squares):
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            eval = minimax(new_lines, squares, depth-1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_valid_moves(lines_drawn, squares):
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 1})
            eval = minimax(new_lines, squares, depth-1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def check_new_squares(new_edge, lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(new_edge) # giả định thêm new_line

    new_squares = []
    for idx, square in enumerate(squares):
        edges = edges_of_square(square)
        if new_edge in edges:
            if all(edge in current_edges for edge in edges):
                new_squares.append(idx)
    
    return new_squares   

def draw_colored_squares(idx, squares, dot_positions, canvas, color=BLACK):
    a, b, c, d = squares[idx]
    points = [dot_positions[a], dot_positions[b], dot_positions[d], dot_positions[c]]   # theo chiều kim đồng hồ
    pygame.draw.polygon(canvas, color, points)

def evaluate_forced_move(move, lines_drawn, squares):
    # Simulate taking the forced move
    new_lines = lines_drawn.copy()
    new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
    
    # Count how many boxes this move completes
    completed_boxes = len(check_new_squares((move[0], move[1]), lines_drawn, squares))
    
    # Look ahead to see if this creates more forced moves for the opponent
    opponent_forced_moves = find_forced_moves(new_lines, squares)
    
    # Calculate a score based on:
    # 1. Number of boxes we complete
    # 2. Number of forced moves we create for opponent (penalty)
    # 3. Whether this move creates a chain reaction
    score = completed_boxes * COMPLETE_SQUARE_VALUE
    score -= len(opponent_forced_moves) * CHAIN_PENALTY
    
    # If this move creates a chain reaction, heavily penalize it
    if len(opponent_forced_moves) > 1:
        score -= CHAIN_PENALTY * 2
    
    return score

def bot_choose_move(lines_drawn, squares):
    # try opening book first
    opening_move = get_opening_move(get_board_size(squares), lines_drawn)
    if opening_move:
        return opening_move

    # check for forced moves (completing squares)
    forced_moves = find_forced_moves(lines_drawn, squares)
    
    # If there are forced moves, evaluate them instead of taking the first one
    if forced_moves:
        best_forced_move = None
        best_forced_score = float('-inf')
        
        for move in forced_moves:
            score = evaluate_forced_move(move, lines_drawn, squares)
            if score > best_forced_score:
                best_forced_score = score
                best_forced_move = move
        
        # Only take the forced move if it's not too detrimental
        if best_forced_score > -CHAIN_PENALTY * 2:
            return best_forced_move
    
    # use minimax for non-forced moves
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in get_valid_moves(lines_drawn, squares):
        new_lines = lines_drawn.copy()
        new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
        score = minimax(new_lines, squares, 3, alpha, beta, False)
        
        # Adjust score based on chain reaction prediction
        chain_length = predict_chain_reaction(move, lines_drawn, squares)
        score -= chain_length * CHAIN_PENALTY
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move

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
                                    current_player = 2  # Chuyển sang bot
                                    
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
                    current_player = 1  # Chuyển lại cho người chơi

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