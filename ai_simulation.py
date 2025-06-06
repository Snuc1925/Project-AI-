import time
import random
import pygame
from pygame import gfxdraw
import os
from main1 import (
    generate_squares, edges_of_square, extract_edge_set, get_valid_moves,
    is_game_over, check_new_squares, bot_choose_move
)
from main_grok import (
    bot_choose_move as grok_bot_choose_move,
    evaluate_position as grok_evaluate_position,
    predict_chain_reaction as grok_predict_chain_reaction,
    find_forced_moves as grok_find_forced_moves,
    get_opening_move as grok_get_opening_move,
    move_priority
)
from αβpruning import (
    bot_choose_move as grok1_bot_choose_move,
    move_priority as grok1_move_priority,
    initialize_zobrist_table,
    TranspositionTable
)

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

# Time management constants
MIN_MOVE_TIME = 1.5
MAX_MOVE_TIME = 7.5
BASE_MOVE_TIME = 4.5
CRITICAL_MOVE_TIME = 7.0
VISUAL_DELAY = 1.0  # seconds between moves for observer
TIME_LIMIT = 5.0  # seconds per move for base AI

# Board size options
BOARD_SIZES = {
    "3x3": (3, 3),
    "4x4": (4, 4),
    "4x5": (4, 5),
    "5x5": (5, 5)
}

class AISimulation:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.squares = generate_squares(rows, cols)
        self.lines_drawn = []
        self.current_player = 1
        self.scores = {1: 0, 2: 0}
        self.completed_squares = {1: set(), 2: set()}
        self.move_times = {1: [], 2: []}
        self.total_moves = 0
        self.move_history = []
        self.chain_count = {1: 0, 2: 0}
        self.three_edge_squares = {1: 0, 2: 0}
        
        # Initialize pygame display and mixer
        pygame.init()
        pygame.mixer.init()
        
        # Load and play background music
        music_path = os.path.join(os.path.dirname(__file__), 'sounds', 'bgm.mp3')
        try:
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)
            else:
                print(f"Warning: Background music file not found at {music_path}")
        except Exception as e:
            print(f"Warning: Could not load background music: {e}")
        
        self.screen_width = (cols - 1) * GRID_SPACING + 2 * MARGIN
        self.screen_height = (rows - 1) * GRID_SPACING + 2 * MARGIN
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Simulation")
        
        try:
            self.font = pygame.font.SysFont('Arial', 30)
        except:
            self.font = pygame.font.Font(None, 40)
            
        # Calculate dot positions
        self.dot_positions = []
        for r in range(rows):
            for c in range(cols):
                x = c * GRID_SPACING + MARGIN
                y = r * GRID_SPACING + MARGIN
                self.dot_positions.append((x, y))
                
        self.clock = pygame.time.Clock()

    def __del__(self):
        # Stop music when the simulation ends
        try:
            pygame.mixer.music.stop()
        except:
            pass

    def _base_ai_move(self):
        start_time = time.time()
        
        # Use the basic alpha-beta pruning implementation from main.py
        move = bot_choose_move(self.lines_drawn, self.squares)
        
        return move

    def _grok_ai_move(self):
        start_time = time.time()
        
        # Get valid moves and sort by priority
        valid_moves = get_valid_moves(self.lines_drawn, self.squares)
        if not valid_moves:
            return None
            
        valid_moves.sort(key=lambda m: move_priority(m, self.lines_drawn, self.squares), reverse=True)
        return grok_bot_choose_move(self.lines_drawn, self.squares)

    def _grok1_ai_move(self):
        start_time = time.time()
        
        # Initialize Zobrist table and transposition table
        initialize_zobrist_table()
        transposition_table = TranspositionTable()
        
        valid_moves = get_valid_moves(self.lines_drawn, self.squares)
        if not valid_moves:
            return None
            
        valid_moves.sort(key=lambda m: grok1_move_priority(m, self.lines_drawn, self.squares), reverse=True)
        return grok1_bot_choose_move(self.lines_drawn, self.squares)

    def make_move(self, player, ai_type):
        start_time = time.time()
        
        if ai_type == "αβpruning":
            move = self._base_ai_move()
        elif ai_type == "αβ_grok":
            move = self._grok_ai_move()
        else:  # grok1
            move = self._grok1_ai_move()
            
        if move:
            id1, id2 = move
            new_squares = check_new_squares((id1, id2), self.lines_drawn, self.squares)
            self.lines_drawn.append({"id1": id1, "id2": id2, "player": player})
            
            # Update three-edge squares count
            current_edges = extract_edge_set(self.lines_drawn)
            self.three_edge_squares[player] = sum(1 for square in self.squares 
                                                if len([e for e in edges_of_square(square) if e in current_edges]) == 3)
            
            if new_squares:
                self.scores[player] += len(new_squares)
                self.completed_squares[player].update(new_squares)
                self.chain_count[player] += 1
            else:
                self.current_player = 3 - player
                
            self.total_moves += 1
            move_time = time.time() - start_time
            self.move_times[player].append(move_time)
            self.move_history.append({
                "id1": id1, 
                "id2": id2, 
                "player": player, 
                "time": move_time,
                "three_edge_squares": self.three_edge_squares[player]
            })
            
        return move

    def get_stats(self):
        return {
            "scores": self.scores,
            "total_moves": self.total_moves,
            "avg_move_time": {
                1: sum(self.move_times[1]) / len(self.move_times[1]) if self.move_times[1] else 0,
                2: sum(self.move_times[2]) / len(self.move_times[2]) if self.move_times[2] else 0
            },
            "chain_moves": self.chain_count,
            "three_edge_squares": self.three_edge_squares,
            "move_history": self.move_history
        }

    def draw_board(self, player1_name, player2_name, show_next_button=False):
        self.screen.fill(WHITE)
        
        # Draw scores and game state
        score_p1_text = f"{player1_name}: {self.scores[1]}"
        score_p2_text = f"{player2_name}: {self.scores[2]}"
        score_p1_surf = self.font.render(score_p1_text, True, BLUE)
        score_p2_surf = self.font.render(score_p2_text, True, RED)
        
        # Draw three-edge squares count
        three_edge_p1_text = f"3-edge: {self.three_edge_squares[1]}"
        three_edge_p2_text = f"3-edge: {self.three_edge_squares[2]}"
        three_edge_p1_surf = self.font.render(three_edge_p1_text, True, BLUE)
        three_edge_p2_surf = self.font.render(three_edge_p2_text, True, RED)
        
        p1_width = score_p1_surf.get_width()
        spacing = 50
        start_x_p1 = (self.screen_width - (p1_width + spacing + score_p2_surf.get_width())) // 2
        start_x_p2 = start_x_p1 + p1_width + spacing
        text_y = 30
        
        self.screen.blit(score_p1_surf, (start_x_p1, text_y))
        self.screen.blit(score_p2_surf, (start_x_p2, text_y))
        self.screen.blit(three_edge_p1_surf, (start_x_p1, text_y + 30))
        self.screen.blit(three_edge_p2_surf, (start_x_p2, text_y + 30))
        
        # Draw squares
        for square_idx in range(len(self.squares)):
            if square_idx in self.completed_squares[1]:
                self.draw_colored_square(square_idx, LIGHT_BLUE)
            elif square_idx in self.completed_squares[2]:
                self.draw_colored_square(square_idx, LIGHT_RED)
                
        # Draw lines
        for line in self.lines_drawn:
            color = BLUE if line["player"] == 1 else RED
            self.draw_line(line["id1"], line["id2"], color)
            
        # Draw dots
        for pos in self.dot_positions:
            x, y = pos
            gfxdraw.filled_circle(self.screen, x, y, DOT_RADIUS, BLACK)
            gfxdraw.aacircle(self.screen, x, y, DOT_RADIUS, BLACK)
            
        # Draw last move indicators
        if self.move_history:
            last_move = self.move_history[-1]
            move_color = BLUE if last_move["player"] == 1 else RED
            pygame.draw.circle(self.screen, move_color, 
                             self.dot_positions[last_move["id1"]], 10)
            pygame.draw.circle(self.screen, move_color, 
                             self.dot_positions[last_move["id2"]], 10)
        
        pygame.display.flip()

    def draw_colored_square(self, idx, color):
        a, b, c, d = self.squares[idx]
        points = [self.dot_positions[a], self.dot_positions[b], 
                 self.dot_positions[d], self.dot_positions[c]]
        pygame.draw.polygon(self.screen, color, points)
        
    def draw_line(self, id1, id2, color):
        if not (0 <= id1 < len(self.dot_positions) and 0 <= id2 < len(self.dot_positions)):
            return
        try:
            vt1 = self.dot_positions[id1]
            vt2 = self.dot_positions[id2]
            pygame.draw.line(self.screen, color, vt1, vt2, THICKNESS)
        except IndexError:
            print(f"Error drawing line: Index out of range for IDs ({id1}, {id2})")

def wait_for_next_match_button(screen, font, screen_width, screen_height):
    button_width, button_height = 150, 40  # Smaller button
    button_x = (screen_width - button_width) // 2
    button_y = screen_height - button_height - 15  # Lower position
    
    # Draw the button
    pygame.draw.rect(screen, GREEN, (button_x, button_y, button_width, button_height), border_radius=12)
    btn_text = font.render("Next Match", True, WHITE)
    text_rect = btn_text.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
    screen.blit(btn_text, text_rect)
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if (button_x <= mx <= button_x + button_width and
                    button_y <= my <= button_y + button_height):
                    return True  # Return True to indicate button was clicked
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Return False to indicate escape was pressed
        pygame.time.wait(20)

def ai_selection_menu(screen_width, screen_height, font):
    pygame.init()
    # Make window slightly smaller
    screen_width = max(screen_width, 800)  # Reduced from 900
    screen_height = max(screen_height, 700)  # Reduced from 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select AIs for Match")
    clock = pygame.time.Clock()
    
    ai_options = ["αβpruning", "αβ_grok", "αβ_grok1"]
    selected = {1: None, 2: None}
    
    # Calculate button width based on longest text
    button_h = 70  # Slightly reduced height
    button_w = max(font.size(ai)[0] for ai in ai_options) + 60  # Add padding for text
    spacing = 40  # Reduced spacing
    top_margin = 120  # Reduced top margin
    left_margin = 120  # Reduced left margin
    
    # Calculate total height needed for AI buttons
    total_buttons_height = len(ai_options) * (button_h + spacing) - spacing
    # Calculate bottom margin to ensure Start button doesn't overlap
    bottom_margin = 120  # Reduced bottom margin
    
    running = True
    while running:
        screen.fill(WHITE)
        # Titles
        p1_label = font.render("Player 1 AI:", True, BLUE)
        p2_label = font.render("Player 2 AI:", True, RED)
        screen.blit(p1_label, (left_margin, top_margin - 50))  # Adjusted title position
        screen.blit(p2_label, (screen_width//2 + left_margin//2, top_margin - 50))
        
        # Draw AI buttons for Player 1
        p1_buttons = []
        for i, ai in enumerate(ai_options):
            x = left_margin
            y = top_margin + i * (button_h + spacing)
            rect = pygame.Rect(x, y, button_w, button_h)
            color = GREEN if selected[1] == ai else BLACK
            pygame.draw.rect(screen, color, rect, border_radius=12, width=0 if selected[1] == ai else 2)
            text = font.render(ai, True, WHITE if selected[1] == ai else color)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            p1_buttons.append((rect, ai))
        
        # Draw AI buttons for Player 2
        p2_buttons = []
        for i, ai in enumerate(ai_options):
            x = screen_width//2 + left_margin//2
            y = top_margin + i * (button_h + spacing)
            rect = pygame.Rect(x, y, button_w, button_h)
            color = GREEN if selected[2] == ai else BLACK
            pygame.draw.rect(screen, color, rect, border_radius=12, width=0 if selected[2] == ai else 2)
            text = font.render(ai, True, WHITE if selected[2] == ai else color)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            p2_buttons.append((rect, ai))
        
        # Draw Start Match button with proper spacing
        start_enabled = selected[1] is not None and selected[2] is not None and selected[1] != selected[2]
        start_color = GREEN if start_enabled else (180, 180, 180)
        start_rect = pygame.Rect((screen_width-250)//2, screen_height-bottom_margin, 250, button_h)  # Adjusted size
        pygame.draw.rect(screen, start_color, start_rect, border_radius=12)
        start_text = font.render("Start Match", True, WHITE)
        start_text_rect = start_text.get_rect(center=start_rect.center)
        screen.blit(start_text, start_text_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for rect, ai in p1_buttons:
                    if rect.collidepoint(mx, my):
                        selected[1] = ai
                for rect, ai in p2_buttons:
                    if rect.collidepoint(mx, my):
                        selected[2] = ai
                if start_rect.collidepoint(mx, my) and start_enabled:
                    return selected[1], selected[2]
        clock.tick(30)

def board_size_selection_menu(screen_width, screen_height, font):
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select Board Size")
    clock = pygame.time.Clock()
    
    title = font.render("Select Board Size", True, BLACK)
    title_rect = title.get_rect(center=(screen_width//2, 50))
    
    button_w, button_h = 160, 60
    spacing = 30
    top_margin = 120
    left_margin = (screen_width - (button_w * 2 + spacing)) // 2
    
    buttons = []
    for i, (size_name, _) in enumerate(BOARD_SIZES.items()):
        row = i // 2
        col = i % 2
        x = left_margin + col * (button_w + spacing)
        y = top_margin + row * (button_h + spacing)
        rect = pygame.Rect(x, y, button_w, button_h)
        buttons.append((rect, size_name))
    
    running = True
    while running:
        screen.fill(WHITE)
        screen.blit(title, title_rect)
        
        for rect, size_name in buttons:
            pygame.draw.rect(screen, BLACK, rect, border_radius=10, width=2)
            text = font.render(size_name, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for rect, size_name in buttons:
                    if rect.collidepoint(mx, my):
                        return BOARD_SIZES[size_name]
        
        clock.tick(30)

def time_mode_selection_menu(screen_width, screen_height, font):
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select Time Mode")
    clock = pygame.time.Clock()
    
    title = font.render("Select Time Mode", True, BLACK)
    title_rect = title.get_rect(center=(screen_width//2, 50))
    
    button_w, button_h = 400, 60
    spacing = 30
    top_margin = 120
    left_margin = (screen_width - button_w) // 2
    
    # Create buttons for each time mode
    human_rect = pygame.Rect(left_margin, top_margin, button_w, button_h)
    fixed_rect = pygame.Rect(left_margin, top_margin + button_h + spacing, button_w, button_h)
    none_rect = pygame.Rect(left_margin, top_margin + 2 * (button_h + spacing), button_w, button_h)
    
    running = True
    while running:
        screen.fill(WHITE)
        screen.blit(title, title_rect)
        
        # Draw buttons
        pygame.draw.rect(screen, BLACK, human_rect, border_radius=10, width=2)
        pygame.draw.rect(screen, BLACK, fixed_rect, border_radius=10, width=2)
        pygame.draw.rect(screen, BLACK, none_rect, border_radius=10, width=2)
        
        # Draw button text
        human_text = font.render("Human-like", True, BLACK)
        fixed_text = font.render("Fixed (7.5 secs/move)", True, BLACK)
        none_text = font.render("Unlimited", True, BLACK)
        
        human_text_rect = human_text.get_rect(center=human_rect.center)
        fixed_text_rect = fixed_text.get_rect(center=fixed_rect.center)
        none_text_rect = none_text.get_rect(center=none_rect.center)
        
        screen.blit(human_text, human_text_rect)
        screen.blit(fixed_text, fixed_text_rect)
        screen.blit(none_text, none_text_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if human_rect.collidepoint(mx, my):
                    return "human"
                elif fixed_rect.collidepoint(mx, my):
                    return "fixed"
                elif none_rect.collidepoint(mx, my):
                    return "none"
        
        clock.tick(60)

def run_simulation(rows, cols, ai1, ai2, max_games=1, swap_colors=True):
    games_played = 0
    while games_played < max_games:
        print(f"\nStarting game {games_played + 1}")
        sim = AISimulation(rows, cols)
        
        # Swap AI positions if needed
        if swap_colors and games_played % 2 == 1:
            ai1, ai2 = ai2, ai1
            
        player1_ai = ai1
        player2_ai = ai2
        
        print(f"Player 1: {player1_ai}")
        print(f"Player 2: {player2_ai}")
        
        # Game loop
        running = True
        while running and not is_game_over(sim.lines_drawn, sim.squares):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return False
                    
            current_ai = player1_ai if sim.current_player == 1 else player2_ai
            move = sim.make_move(sim.current_player, current_ai)
            
            if not move:
                break
                
            # Update display
            sim.draw_board(player1_ai, player2_ai)
            sim.clock.tick(60)
            time.sleep(VISUAL_DELAY)
            
        stats = sim.get_stats()
        print(f"Final scores: {stats['scores']}")
        print(f"Total moves: {stats['total_moves']}")
        print(f"Average move times: {stats['avg_move_time']}")
        print(f"Chain moves: {stats['chain_moves']}")
        print(f"Three-edge squares: {stats['three_edge_squares']}")
        
        # Show final board
        sim.draw_board(player1_ai, player2_ai)
        
        # Show Next Match button and wait for click
        if not wait_for_next_match_button(sim.screen, sim.font, sim.screen_width, sim.screen_height):
            return False
            
        # Update results
        p1_score = stats['scores'][1]
        p2_score = stats['scores'][2]
        
        print(f"\nGame Results:")
        print(f"{player1_ai}: {p1_score} points")
        print(f"{player2_ai}: {p2_score} points")
        if p1_score > p2_score:
            print(f"{player1_ai} wins!")
        elif p2_score > p1_score:
            print(f"{player2_ai} wins!")
        else:
            print("It's a tie!")
        
        games_played += 1
        pygame.quit()
        
    return True

def main():
    while True:
        # Menu for board size selection
        pygame.init()
        menu_width, menu_height = 700, 400
        try:
            menu_font = pygame.font.SysFont('Arial', 32)
        except:
            menu_font = pygame.font.Font(None, 40)
            
        # Get board size first
        board_size = board_size_selection_menu(menu_width, menu_height, menu_font)
        if board_size is None:
            break
            
        rows, cols = board_size
        
        # Then get AI selection
        ai1, ai2 = ai_selection_menu(menu_width, menu_height, menu_font)
        if ai1 is None or ai2 is None:
            break
            
        # Run simulation with selected board size and AIs
        if not run_simulation(rows, cols, ai1=ai1, ai2=ai2, max_games=1, swap_colors=True):
            break
            
        pygame.quit()

if __name__ == "__main__":
    main()