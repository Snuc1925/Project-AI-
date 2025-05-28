import time
import random
import pygame
from pygame import gfxdraw
import os
from main import (
    generate_squares, edges_of_square, extract_edge_set, get_valid_moves,
    is_game_over, get_board_size, check_new_squares, find_forced_moves,
    get_opening_move, minimax, evaluate_position, predict_chain_reaction
)
from main_grok import (
    bot_choose_move,
    evaluate_position as grok_evaluate_position,
    predict_chain_reaction as grok_predict_chain_reaction,
    find_forced_moves as grok_find_forced_moves,
    get_opening_move as grok_get_opening_move,
    move_priority, evaluate_forced_move
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

VISUAL_DELAY = 2  # seconds between moves for observer
TIME_LIMIT = 5.0  # seconds per move

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
        self.game_phase = "opening"
        self.chain_count = {1: 0, 2: 0}
        self.three_edge_squares = {1: 0, 2: 0}  # Track three-edge squares for each player
        
        # Initialize pygame display and mixer
        pygame.init()
        pygame.mixer.init()
        
        # Load and play background music
        music_path = os.path.join(os.path.dirname(__file__), 'sounds', 'bgm.mp3')
        try:
            if os.path.exists(music_path):
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)  # Set volume to 50%
                pygame.mixer.music.play(-1)  # -1 means loop indefinitely
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
        
        # Try opening book first
        opening_move = get_opening_move(get_board_size(self.squares), self.lines_drawn)
        if opening_move:
            return opening_move
            
        # Check for forced moves
        forced_moves = find_forced_moves(self.lines_drawn, self.squares)
        if forced_moves:
            return forced_moves[0]
            
        # Use minimax for non-forced moves
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in get_valid_moves(self.lines_drawn, self.squares):
            if time.time() - start_time > TIME_LIMIT:
                break
                
            new_lines = self.lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            score = minimax(new_lines, self.squares, 3, alpha, beta, False)
            chain_length = predict_chain_reaction(move, self.lines_drawn, self.squares)
            score -= chain_length * 4.5
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def _improved_ai_move(self):
        start_time = time.time()
        
        # Try opening book first
        opening_move = get_opening_move(get_board_size(self.squares), self.lines_drawn)
        if opening_move:
            return opening_move
            
        # Check for forced moves
        forced_moves = find_forced_moves(self.lines_drawn, self.squares)
        if forced_moves:
            best_forced_move = None
            best_forced_score = float('-inf')
            
            for move in forced_moves:
                if time.time() - start_time > TIME_LIMIT:
                    break
                    
                score = evaluate_forced_move(move, self.lines_drawn, self.squares)
                if score > best_forced_score:
                    best_forced_score = score
                    best_forced_move = move
                    
            if best_forced_score > -9:  # CHAIN_PENALTY_BASE * 2
                return best_forced_move
                
        return self._base_ai_move()

    def _grok_ai_move(self):
        start_time = time.time()
        
        # Count three-edge squares for time management
        current_edges = extract_edge_set(self.lines_drawn)
        three_edge_count = sum(1 for square in self.squares 
                             if len([e for e in edges_of_square(square) if e in current_edges]) == 3)
        
        # Determine time limit based on game state
        if three_edge_count >= 2:  # Critical position
            time_limit = 7.0  # CRITICAL_MOVE_TIME
        elif three_edge_count == 1:  # Single three-edge square
            time_limit = 5.5  # BASE_MOVE_TIME + 1.0
        else:
            time_limit = 4.5  # BASE_MOVE_TIME
            
        # Check for opening moves
        if len(self.lines_drawn) < 2:
            time.sleep(1.5)  # MIN_MOVE_TIME for opening moves
            
        move = bot_choose_move(self.lines_drawn, self.squares)
        
        # Ensure minimum move time
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.5:  # MIN_MOVE_TIME
            time.sleep(1.5 - elapsed_time)
            
        return move

    def _update_game_phase(self):
        total_squares = len(self.squares)
        completed = len(self.completed_squares[1]) + len(self.completed_squares[2])
        completion_ratio = completed / total_squares if total_squares > 0 else 0
        
        if completion_ratio < 0.25:
            self.game_phase = "opening"
        elif completion_ratio < 0.75:
            self.game_phase = "midgame"
        else:
            self.game_phase = "endgame"

    def make_move(self, player, ai_type):
        start_time = time.time()
        
        if ai_type == "base":
            move = self._base_ai_move()
        elif ai_type == "improved":
            move = self._improved_ai_move()
        else:  # grok
            move = self._grok_ai_move()
            
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
            
            if move_time > 7.5:  # MAX_MOVE_TIME
                print(f"Warning: {ai_type} AI exceeded time limit ({move_time:.2f}s)")
                
            self._update_game_phase()
            
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
            "game_phase": self.game_phase,
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
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Select AIs for Match")
    clock = pygame.time.Clock()
    
    ai_options = ["base", "improved", "grok"]
    selected = {1: None, 2: None}
    button_w, button_h = 160, 60
    spacing = 30
    top_margin = 80
    left_margin = 80
    
    running = True
    while running:
        screen.fill(WHITE)
        # Titles
        p1_label = font.render("Player 1 AI:", True, BLUE)
        p2_label = font.render("Player 2 AI:", True, RED)
        screen.blit(p1_label, (left_margin, top_margin - 50))
        screen.blit(p2_label, (screen_width//2 + left_margin//2, top_margin - 50))
        
        # Draw AI buttons for Player 1
        p1_buttons = []
        for i, ai in enumerate(ai_options):
            x = left_margin
            y = top_margin + i * (button_h + spacing)
            rect = pygame.Rect(x, y, button_w, button_h)
            color = GREEN if selected[1] == ai else BLACK
            pygame.draw.rect(screen, color, rect, border_radius=10, width=0 if selected[1] == ai else 2)
            text = font.render(ai.capitalize(), True, WHITE if selected[1] == ai else color)
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
            pygame.draw.rect(screen, color, rect, border_radius=10, width=0 if selected[2] == ai else 2)
            text = font.render(ai.capitalize(), True, WHITE if selected[2] == ai else color)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
            p2_buttons.append((rect, ai))
        
        # Draw Start Match button
        start_enabled = selected[1] is not None and selected[2] is not None and selected[1] != selected[2]
        start_color = GREEN if start_enabled else (180, 180, 180)
        start_rect = pygame.Rect((screen_width-200)//2, screen_height-100, 200, 60)
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

def run_simulation(rows, cols, ai1="base", ai2="improved", max_games=1, swap_colors=True):
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
        print(f"Game phase: {stats['game_phase']}")
        
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
        # Menu for AI selection
        pygame.init()
        menu_width, menu_height = 700, 400
        try:
            menu_font = pygame.font.SysFont('Arial', 32)
        except:
            menu_font = pygame.font.Font(None, 40)
            
        ai1, ai2 = ai_selection_menu(menu_width, menu_height, menu_font)
        if ai1 is None or ai2 is None:
            break
            
        # Run simulation with selected AIs
        if not run_simulation(5, 5, ai1=ai1, ai2=ai2, max_games=1, swap_colors=True):
            break
            
        pygame.quit()

if __name__ == "__main__":
    main()