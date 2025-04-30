import pygame
import sys
from pygame import gfxdraw
import os


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
DOT_RADIUS = 5
GRID_SPACING = 100
MARGIN = 100
CLICK_RADIUS = 15
LINE_COLOR = BLACK
DOT_HIGHLIGHT_RADIUS = 7
PLAYER1_COLOR = BLUE
PLAYER2_COLOR = RED


### Kiểm tra chọn điểm
def get_point_at_vt(dot_positions, x, y, click_radius=CLICK_RADIUS):
    for i, vt in enumerate(dot_positions):
        px, py = vt
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist < click_radius ** 2:
            return i
    return None


### Vẽ nét
def draw_line(surface, dot_positions, id1, id2, color, thickness=3):
    num_dots = len(dot_positions)
    if not (0 <= id1 < num_dots and 0 <= id2 < num_dots): return
    try:
        vt1 = dot_positions[id1]
        vt2 = dot_positions[id2]
        pygame.draw.line(surface, color, vt1, vt2, thickness)
    except IndexError:
        print(f"Error drawing line: Index out of range for IDs ({id1}, {id2})")


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

def check_line(id1, id2, dot_positions):
    is_adjacent = False
    vt1 = dot_positions[id1]
    vt2 = dot_positions[id2]

    # Kiểm tra chiều ngang (cùng y, khác biệt x bằng GRID_SPACING)
    if vt1[1] == vt2[1] and abs(vt1[0] - vt2[0]) == GRID_SPACING:
        is_adjacent = True

    # Kiểm tra chiều dọc (cùng x, khác biệt y bằng GRID_SPACING)
    elif vt1[0] == vt2[0] and abs(vt1[1] - vt2[1]) == GRID_SPACING:
        is_adjacent = True

    return is_adjacent



########################################################################################################################################################
########################################################################################################################################################

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

    icon_path = "dotsandboxes.png"
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

    point1 = 0
    point2 = 0
    score_p1_text = f"{player1_name}: {point1}"
    score_p2_text = f"{player2_name}: {point2}"

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
    boxes_drawn = []
    for i, dot in enumerate(dot_positions):
        num_dots = len(dot_positions)
        if (dot[0] + GRID_SPACING <= (cols - 1) * GRID_SPACING + MARGIN) and (dot[1] + GRID_SPACING <= (rows - 1) * GRID_SPACING + MARGIN):
            boxes_drawn.append({"id1": i,
                                "id2": i + 1,
                                "id3": i + 1 + cols,
                                "id4": i + cols,
                                "color": WHITE,
                                "num_lines": 0})
    #############################
    print("dot_positions:", dot_positions)
    print("boxes_drawn:", boxes_drawn)
    print("cols:", cols)
    print("rows:", rows)

    running = True

    player = 1

    while running:

        state = {
            "dot_positions": dot_positions,
            "lines_drawn": lines_drawn,
            "boxes_drawn": boxes_drawn,
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if (player == 1):
                color = PLAYER1_COLOR
                if event.type == pygame.MOUSEBUTTONDOWN:
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

                            is_adjacent = check_line(id1, id2, dot_positions)

                            if is_adjacent:
                                id_pair = tuple(sorted((id1, id2)))
                                line_exists = any(
                                    line['id1'] == id_pair[0] and line['id2'] == id_pair[1] for line in lines_drawn)

                                if not line_exists:
                                    lines_drawn.append({"id1": id_pair[0], "id2": id_pair[1], "color": color})
                                    print("boxes_drawn:", boxes_drawn)

                                    ### Thay đổi trạng thái ô
                                    check_box(dot_positions, lines_drawn, boxes_drawn, color)
                                    state = {
                                        "dot_positions": dot_positions.copy(),
                                        "lines_drawn": lines_drawn.copy(),
                                        "boxes_drawn": boxes_drawn.copy(),
                                    }

                                    player = 2
                                else:
                                    print("Line already exists.")
                            else:
                                print("Dots are not adjacent, line not added.")

                            selected_dot_id = None

                    else:
                        print("Clicked empty space, deselecting.")
                        selected_dot_id = None
            else:
                color = PLAYER2_COLOR
                #print("old_state:", len(state["lines_drawn"]))
                new_state = move_for_ai(state, 3)
                #print("new_state:", len(new_state["lines_drawn"]))
                lines_drawn = new_state["lines_drawn"]
                boxes_drawn = new_state["boxes_drawn"]
                #print("new_state2:", len(lines_drawn))
                #print("new_state2['lines_drawn']: ", lines_drawn)
                #print("new_state2['boxes_drawn']: ", boxes_drawn)
                check_box(dot_positions, lines_drawn, boxes_drawn, color)
                #print("After main AI check_box")
                player = 1

        SURF.fill(WHITE)

        SURF.blit(score_p1_surf, (start_x_p1, text_y))
        SURF.blit(score_p2_surf, (start_x_p2, text_y))

######## Vẽ đuường
        for line_info in lines_drawn:
            draw_line(SURF, dot_positions, line_info["id1"], line_info["id2"], line_info["color"])


######## Cập nhật điểm
        p1, p2 = check_point(boxes_drawn)
        point1 = p1
        point2 = p2
        score_p1_text = f"{player1_name}: {point1}"
        score_p2_text = f"{player2_name}: {point2}"
        score_p1_surf = score_font.render(score_p1_text, True, player1_color)
        score_p2_surf = score_font.render(score_p2_text, True, player2_color)

######## Vẽ ô
        for box_info in boxes_drawn:
            draw_box(SURF, dot_positions, box_info["id1"], box_info["id2"], box_info["id3"], box_info["id4"], box_info["color"])

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


def draw_box(surface, dot_positions, id1, id2, id3, id4, color, thickness=3, inner_margin=5):
    # Kiểm tra tính hợp lệ của các chỉ số id1, id2, id3, id4 trong danh sách dot_positions
    num_dots = len(dot_positions)
    if not all(0 <= id < num_dots for id in [id1, id2, id3, id4]):
        return  # Nếu có chỉ số không hợp lệ thì không làm gì cả

    try:
        # Lấy tọa độ của bốn điểm để tạo thành một ô vuông
        vt1 = dot_positions[id1]
        vt2 = dot_positions[id2]
        vt3 = dot_positions[id3]
        vt4 = dot_positions[id4]

        # Tính toán tọa độ để tạo khoảng cách bên trong (inner_margin)
        x1, y1 = vt1
        x2, y2 = vt3  # Góc đối diện để lấy kích thước ô vuông

        # Tính toán tọa độ của ô vuông bên trong (có khoảng cách margin)
        x1_inner = x1 + inner_margin
        y1_inner = y1 + inner_margin
        x2_inner = x2 - inner_margin
        y2_inner = y2 - inner_margin

        # Tô màu nền của ô vuông
        pygame.draw.rect(surface, color, (x1_inner, y1_inner, x2_inner - x1_inner, y2_inner - y1_inner))  # Tô màu bên trong

    except IndexError:
        print(f"Error drawing box: Index out of range for IDs ({id1}, {id2}, {id3}, {id4})")


def check_box(dot_positions, lines_drawn, boxes_drawn, color):
    # print("Start main AI check_box")
    # print("Before loop")
    # print("lines_drawn color: ", len(lines_drawn))
    # print("lines_drawn in color: ", lines_drawn)
    # print("boxes_drawn in color: ", boxes_drawn)
    # print("After loop")
    # print("color: ", color)
    latest_line = lines_drawn[-1]
    for box_info in boxes_drawn:
        id1 = box_info["id1"]
        id2 = box_info["id2"]
        id3 = box_info["id3"]
        id4 = box_info["id4"]

        # Các cặp điểm tạo thành các cạnh của ô
        lines = [
            (id1, id2),
            (id2, id3),
            (id4, id3),
            (id1, id4)
        ]

        # Kiểm tra xem tất cả các đường nối đã được vẽ hay chưa
        all_lines_exist = all(
            any(line['id1'] == min(pair) and line['id2'] == max(pair) for line in lines_drawn)
            for pair in lines
        )

        # Nếu tất cả các cạnh đã vẽ
        if all_lines_exist:
            # print("lines_drawn color: ", len(lines_drawn))
            # print("lines_drawn in color: ", lines_drawn)
            # print("boxes_drawn in color: ", boxes_drawn)
            if tuple(sorted([latest_line["id1"], latest_line["id2"]])) in [tuple(sorted([pair[0], pair[1]])) for pair in lines]:
                box_info["color"] = color
                # print("final color: ", color, lines)
                # print("lines_drawn color: ", len(lines_drawn))
                # print("lines_drawn in color: ", lines_drawn)
                # print("boxes_drawn in color: ", boxes_drawn)
            box_info["num_lines"] = 4  # Ô hoàn thành 4 cạnh
        else:
            box_info["color"] = WHITE
            # Nếu chưa đủ 4 cạnh, cập nhật num_lines theo số đường đã vẽ
            lines_completed = 0
            for pair in lines:
                line_exists = False
                for line in lines_drawn:
                    if line['id1'] == min(pair) and line['id2'] == max(pair):
                        line_exists = True

                    if line_exists:
                        lines_completed += 1
                        break

            # Cập nhật số lượng đường đã vẽ
            box_info["num_lines"] = lines_completed


##########################################################################################################################################
############################################################ Minimax #####################################################################################################################

def check_point(boxes_drawn):
    point1 = 0
    point2 = 0
    for box_info in boxes_drawn:
        color = box_info["color"]
        if color == PLAYER1_COLOR:
            point1 += 1
        elif color == PLAYER2_COLOR:
            point2 += 1
    return (point1, point2)

def evaluate(state, player):
    point1, point2 = check_point(state["boxes_drawn"])

    return point1 - point2 if player == 1 else point2 - point1

def game_over(state):
    for box_info in state["boxes_drawn"]:
        if (box_info["num_lines"] < 4):
            return False
    return True

def get_possible_lines(state):
    possible_lines = []

    line_drawn = state["lines_drawn"].copy()
    box_drawn = state["boxes_drawn"].copy()
    dot_positions = state["dot_positions"].copy()
    new_state = {
        "dot_positions": dot_positions,
        "lines_drawn": line_drawn,
        "boxes_drawn": box_drawn,
    }

    for i in range(len(dot_positions)):
        for j in range(i+1, len(new_state["dot_positions"])):
            if (check_line(i, j, new_state["dot_positions"])):
                line_exists = any(line['id1'] == i and line['id2'] == j or line['id1'] == j and line['id2'] == i
                                  for line in new_state["lines_drawn"])
                if not line_exists:
                    (i, j) = sorted((i, j))
                    possible_lines.append({"id1": i, "id2": j})

    return possible_lines

def generate_moves(state, player):
    moves = []

    line_drawn = state["lines_drawn"].copy()
    box_drawn = state["boxes_drawn"].copy()
    dot_positions = state["dot_positions"].copy()
    new_state = {
        "dot_positions": dot_positions,
        "lines_drawn": line_drawn,
        "boxes_drawn": box_drawn,
    }

    possible_lines = get_possible_lines(new_state)
    color = PLAYER1_COLOR if player == 1 else PLAYER2_COLOR


    for line in possible_lines:
        line_drawn = state["lines_drawn"].copy()
        box_drawn = state["boxes_drawn"].copy()
        dot_positions = state["dot_positions"].copy()

        line_drawn.append({"id1": line["id1"], "id2": line["id2"], "color": color})
        check_box(dot_positions, line_drawn, box_drawn, color)

        next_state = {
            "dot_positions": dot_positions,
            "lines_drawn": line_drawn,
            "boxes_drawn": box_drawn,
        }
        moves.append(next_state)

    return moves


def minimax(state, depth, alpha, beta, maximizing_player):

    dot_positions = state["dot_positions"].copy()
    lines_drawn = state["lines_drawn"].copy()
    boxes_drawn = state["boxes_drawn"].copy()
    new_state = {
        "dot_positions": dot_positions,
        "lines_drawn": lines_drawn,
        "boxes_drawn": boxes_drawn,
    }

    if (depth == 0) or (game_over(state)):
        return evaluate(state, 2 if maximizing_player else 1)

    if maximizing_player:
        max_eval = float("-inf")
        for next_state in generate_moves(new_state, 2):
            eval = minimax(next_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for next_state in generate_moves(new_state, 1):
            eval = minimax(next_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def move_for_ai(state, depth = 3):
    best_score = float("-inf")
    best_state = None

    dot_positions = state["dot_positions"].copy()
    lines_drawn = state["lines_drawn"].copy()
    boxes_drawn = state["boxes_drawn"].copy()
    new_state = {
        "dot_positions": dot_positions,
        "lines_drawn": lines_drawn,
        "boxes_drawn": boxes_drawn,
    }

    for next_state in generate_moves(new_state, 2):
        score = minimax(next_state, depth, float("-inf"), float("inf"), False)

        if score > best_score:
            best_score = score
            best_state = next_state

    return best_state
