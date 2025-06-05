import pygame
import sys
from pygame import gfxdraw 
import os
from collections import defaultdict
import random

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

def draw_colored_squares(idx, squares, dot_positions, canvas, color=BLACK):
    a, b, c, d = squares[idx]
    points = [dot_positions[a], dot_positions[b], dot_positions[d], dot_positions[c]]  # theo chiều kim đồng hồ
    pygame.draw.polygon(canvas, color, points)

def bot_choose_move(lines_drawn, squares):
    line_set = set((l['id1'], l['id2']) for l in lines_drawn)

    # Nhóm theo số cạnh đã vẽ
    moves_by_edges = {i: [] for i in range(4)}

    for square in squares:
        edges = [
            (square[0], square[1]),  # top
            (square[1], square[3]),  # right
            (square[2], square[3]),  # bottom
            (square[0], square[2])   # left
        ]
        existing = [edge for edge in edges if edge in line_set]
        missing = [edge for edge in edges if edge not in line_set]
        num_existing = len(existing)

        if missing:
            # Ưu tiên 3 cạnh (điền cạnh cuối tạo điểm), sau đó 0,1, rồi 2 cạnh cuối cùng
            moves_by_edges[num_existing].extend(missing)

    if moves_by_edges[3]:  # tạo điểm cho bot
        return random.choice(moves_by_edges[3])
    elif moves_by_edges[0] or moves_by_edges[1]:  # nước an toàn
        safe_moves = moves_by_edges[0] + moves_by_edges[1]
        return random.choice(safe_moves)
    elif moves_by_edges[2]:  # nước nguy hiểm (tạo cơ hội cho đối thủ)
        return random.choice(moves_by_edges[2])
    
    return None  # không còn nước nào


def display_dots(rows, cols, mode, bot1, bot2, on_back_to_menu=None): 
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

    if mode == "AI vs AI":
        is_player1_human = False
    else:
        is_player1_human = True

    is_player2_human = False

    print(mode, is_player1_human, is_player2_human)

    while running:
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

        # Vẽ dots
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

        clock.tick(30)

    pygame.quit()
    

def start_display(board_size_str, mode, bot1, bot2, on_back_to_menu=None):
    try:
        parts = board_size_str.split('x')
        if len(parts) != 2:
            raise ValueError("Board size string must be in 'RowsxCols' format (e.g., '4x5')")

        rows = int(parts[0])
        cols = int(parts[1])

        if rows < 1 or cols < 1:
             print("Error: Board dimensions must be at least 1x1.")
             return

        display_dots(rows, cols, mode, bot1, bot2, on_back_to_menu)

    except (ValueError, IndexError) as e:
        print(f"Error parsing board size string '{board_size_str}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
