import pygame
import sys
from pygame import gfxdraw 
import os
from collections import defaultdict
import random
import copy
import random
import time

sys.setrecursionlimit(10000)    
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

def check_new_squares(new_edge, lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(new_edge)  # giả định thêm new_line

    new_squares = []
    for idx, square in enumerate(squares):
        edges = edges_of_square(square)
        if new_edge in edges:
            if all(edge in current_edges for edge in edges):
                new_squares.append(idx)
    
    return new_squares   

# ----- Test medium
def is_dangerous_move(new_edge, lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(new_edge)

    for square in squares:
        edges = edges_of_square(square)
        cnt = sum(1 for edge in edges if edge in current_edges)
        if cnt == 3 and new_edge in edges:
            return True
    return False

def bot_choose_best_move(lines_drawn, squares):
    line_set = extract_edge_set(lines_drawn)

    candidates = []
    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in line_set]
        missing = [edge for edge in edges if edge not in line_set]

        for edge in missing:
            candidates.append(tuple(sorted(edge)))

    best_move = None
    min_gain = float('inf')

    for move in candidates:
        temp_lines = lines_drawn.copy()
        temp_lines.append({"id1": move[0], "id2": move[1], "player": 0})

        current_edges = extract_edge_set(temp_lines)
        gain = 0

        for square in squares:
            edges = edges_of_square(square)
            if all(edge in current_edges for edge in edges):
                gain += 1

        if gain < min_gain:
            min_gain = gain
            best_move = move

    return best_move


def simulate_player_chain_gain(temp_lines, squares):
    gain = 0
    edges = extract_edge_set(temp_lines)

    while True:
        found = False
        for square in squares:
            sq_edges = edges_of_square(square)
            existing = [e for e in sq_edges if e in edges]
            missing = [e for e in sq_edges if e not in edges]
            if len(existing) == 3:
                move = missing[0]
                edges.add(move)
                temp_lines.append({"id1": move[0], "id2": move[1], "player": 1})
                gain += 1
                found = True
                break 
        if not found:
            break
    return gain


def check_new_squares(new_edge, lines_drawn, squares):
    current_edges = extract_edge_set(lines_drawn)
    current_edges.add(new_edge)  # giả định thêm new_line

    new_squares = []
    for idx, square in enumerate(squares):
        edges = edges_of_square(square)
        if new_edge in edges:
            if all(edge in current_edges for edge in edges):
                new_squares.append(idx)
    
    return new_squares   
# -----


def draw_colored_squares(idx, squares, dot_positions, canvas, color=BLACK):
    a, b, c, d = squares[idx]
    points = [dot_positions[a], dot_positions[b], dot_positions[d], dot_positions[c]]  # theo chiều kim đồng hồ
    pygame.draw.polygon(canvas, color, points)

def bot_choose_move(lines_drawn, squares):
    line_set = extract_edge_set(lines_drawn)

    moves_by_edges = {i: [] for i in range(4)}
    candidates = []

    for square in squares:
        edges = edges_of_square(square)
        existing = [edge for edge in edges if edge in line_set]
        missing = [edge for edge in edges if edge not in line_set]
        num_existing = len(existing)

        for edge in missing:
            candidates.append(tuple(sorted(edge)))

        if missing:
            moves_by_edges[num_existing].extend([tuple(sorted(edge)) for edge in missing])

    if moves_by_edges[3]:
        return random.choice(moves_by_edges[3])

    safe_moves = []
    for move in moves_by_edges[0] + moves_by_edges[1]:
        if not is_dangerous_move(move, lines_drawn, squares):
            safe_moves.append(move)

    if safe_moves:
        return random.choice(safe_moves)

    return bot_choose_best_move(lines_drawn, squares)



def bot_choose_move_MCTS(lines_drawn, squares, list_squares_1=[], list_squares_2=[], time_limit=1.0):
    start_time = time.time()

    all_dots = set()
    for square in squares:
        all_dots.update(square)
    max_dot_id = max(all_dots)

    # 1. Generate all possible valid lines
    possible_moves = []
    for square in squares:
        edges = [
            tuple(sorted((square[0], square[1]))),  # top
            tuple(sorted((square[1], square[3]))),  # right
            tuple(sorted((square[2], square[3]))),  # bottom
            tuple(sorted((square[0], square[2])))   # left
        ]
        for edge in edges:
            if not any(d['id1'] == edge[0] and d['id2'] == edge[1] for d in lines_drawn):
                if edge not in possible_moves:
                    possible_moves.append(edge)

    # NEW: Ưu tiên nước đi tạo thành hình vuông
    for move in possible_moves:
        temp_lines = lines_drawn + [{"id1": move[0], "id2": move[1], "player": 2}]
        if check_new_squares(move, temp_lines, squares):
            return move

    # Giữ lại chỉ các nước đi không nguy hiểm
    safe_moves = [m for m in possible_moves if not is_dangerous_move(m, lines_drawn, squares)]

    if not safe_moves:
        return random.choice(possible_moves)

    possible_moves = safe_moves

    if not possible_moves:
        return None

    move_stats = {move: {"wins": 0, "sims": 0} for move in possible_moves}

    def simulate_game(move):
        # copy trạng thái
        lines_copy = copy.deepcopy(lines_drawn)
        squares_1 = list(list_squares_1)
        squares_2 = list(list_squares_2)

        current = 2  # Bot

        lines_copy.append({"id1": move[0], "id2": move[1], "player": current})
        new_squares = check_new_squares(move, lines_copy, squares)
        if new_squares:
            squares_2.extend(new_squares)
        else:
            current = 1

        total_edges = []
        for square in squares:
            for edge in edges_of_square(square):
                if edge != move and edge not in total_edges:
                    total_edges.append(edge)

        while len(lines_copy) < len(total_edges):
            next_move = bot_choose_move(lines_copy, squares)
            if not next_move:
                break
            lines_copy.append({"id1": next_move[0], "id2": next_move[1], "player": current})
            new_squares = check_new_squares(next_move, lines_copy, squares)
            if new_squares:
                if current == 1:
                    squares_1.extend(new_squares)
                else:
                    squares_2.extend(new_squares)
            else:
                current = 3 - current

        return 2 if len(squares_2) > len(squares_1) else 1

    # 2. MCTS mô phỏng
    while time.time() - start_time < time_limit:
        move = random.choice(possible_moves)
        winner = simulate_game(move)
        move_stats[move]["sims"] += 1
        if winner == 2:
            move_stats[move]["wins"] += 1

    # 3. Chọn nước đi tốt nhất
    best_move = max(possible_moves, key=lambda m: move_stats[m]["wins"] / (move_stats[m]["sims"] + 1e-5))
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
        if len(lines_drawn) == (rows - 1) * cols + rows * (cols - 1):
            print("END GAME!")
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Bot player 1
        if current_player == 1:
            time.sleep(1)
            move = bot_choose_move(lines_drawn, squares)
            if move:
                id1, id2 = move
                new_squares_indices = check_new_squares((id1, id2), lines_drawn, squares)
                lines_drawn.append({"id1": id1, "id2": id2, "player": 1})
                print(f"Bot 1 added line: {id1}-{id2}")

                if new_squares_indices:
                    list_squares_1.extend(new_squares_indices)
                else:
                    current_player = 2

        # Bot player 2
        elif current_player == 2:
            time.sleep(1)
            move = bot_choose_move_MCTS(lines_drawn, squares, list_squares_1, list_squares_2)
            if move:
                id1, id2 = move
                new_squares_indices = check_new_squares((id1, id2), lines_drawn, squares)
                lines_drawn.append({"id1": id1, "id2": id2, "player": 2})
                print(f"Bot 2 (MCTS) added line: {id1}-{id2}")

                if new_squares_indices:
                    list_squares_2.extend(new_squares_indices)
                else:
                    current_player = 1

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
