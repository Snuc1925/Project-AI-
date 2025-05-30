import time
import random
from main import (
    generate_squares, edges_of_square, extract_edge_set, get_valid_moves,
    is_game_over, get_board_size, check_new_squares, find_forced_moves,
    get_opening_move, minimax, evaluate_position, predict_chain_reaction,
    is_square_controlled, bot_choose_move
)
from main_grok import (
    bot_choose_move as grok_bot_choose_move,
    evaluate_position as grok_evaluate_position,
    predict_chain_reaction as grok_predict_chain_reaction,
    find_forced_moves as grok_find_forced_moves,
    get_opening_move as grok_get_opening_move,
    move_priority, evaluate_forced_move
)
from main_mcts import (
    bot_choose_move as mcts_bot_choose_move,
    initialize_zobrist_table,
    compute_zobrist_hash,
    store_transposition,
    lookup_transposition,
    MCTSNode,
    mcts_search,
    mcts_simulation,
    simulation_policy,
    move_priority as mcts_move_priority,
    evaluate_forced_move as mcts_evaluate_forced_move,
    predict_chain_reaction as mcts_predict_chain_reaction,
    evaluate_position as mcts_evaluate_position,
    transposition_table,
    TABLE_SIZE
)

# Board size options
BOARD_SIZES = {
    "3x3": (3, 3),
    "4x4": (4, 4),
    "4x5": (4, 5),
    "5x5": (5, 5)
}

class FastSimulation:
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
        self.three_edge_squares = {1: 0, 2: 0}
        
        # Initialize Zobrist table for MCTS
        self.zobrist_table = initialize_zobrist_table(rows * cols)
        
        # Clear transposition table if it gets too large
        if len(transposition_table) > TABLE_SIZE:
            transposition_table.clear()

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
            if time.time() - start_time > 5.0:  # TIME_LIMIT
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
                if time.time() - start_time > 5.0:  # TIME_LIMIT
                    break
                    
                # Evaluate forced move with chain reaction prediction
                score = evaluate_forced_move(move, self.lines_drawn, self.squares)
                if score > best_forced_score:
                    best_forced_score = score
                    best_forced_move = move
                    
            # Adjust threshold based on game phase
            threshold = -6.0 if self.game_phase == "endgame" else -9.0
            if best_forced_score > threshold:
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
            
        move = grok_bot_choose_move(self.lines_drawn, self.squares)
        
        # Ensure minimum move time
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.5:  # MIN_MOVE_TIME
            time.sleep(1.5 - elapsed_time)
            
        return move

    def _mcts_ai_move(self):
        start_time = time.time()
        
        # Count three-edge squares for time management
        current_edges = extract_edge_set(self.lines_drawn)
        three_edge_count = sum(1 for square in self.squares 
                             if len([e for e in edges_of_square(square) if e in current_edges]) == 3)
        
        # Calculate game phase for time management
        total_squares = len(self.squares)
        completed_squares = sum(1 for square in self.squares if is_square_controlled(square, current_edges))
        game_phase = completed_squares / total_squares if total_squares > 0 else 0
        
        # Determine time limit based on game state
        if game_phase > 0.7:  # Endgame
            time_limit = 7.0  # CRITICAL_MOVE_TIME
        elif three_edge_count >= 2:  # Critical position
            time_limit = 7.0  # CRITICAL_MOVE_TIME
        elif three_edge_count == 1:  # Single three-edge square
            time_limit = 5.5  # BASE_MOVE_TIME + 1.0
        else:
            time_limit = 4.5  # BASE_MOVE_TIME
            
        # Check for opening moves
        if len(self.lines_drawn) < 2:
            time.sleep(1.5)  # MIN_MOVE_TIME
            
        # Initialize Zobrist table if not already done
        if not hasattr(self, 'zobrist_table'):
            board_size = get_board_size(self.squares)
            num_dots = board_size[0] * board_size[1]
            self.zobrist_table = initialize_zobrist_table(num_dots)
            
        # Clear transposition table if it gets too large
        if len(transposition_table) > TABLE_SIZE:
            transposition_table.clear()
            
        # Use MCTS search directly with all required parameters
        best_move = mcts_search(self.lines_drawn, self.squares, self.zobrist_table, time_limit, start_time)
        
        
        # Ensure minimum move time
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.5:  # MIN_MOVE_TIME
            time.sleep(1.5 - elapsed_time)
            
        return best_move

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
        
        if ai_type == "αβpruning":
            move = self._base_ai_move()
        elif ai_type == "αβpruning_improved":
            move = self._improved_ai_move()
        elif ai_type == "MCTS":
            move = self._mcts_ai_move()
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

def get_board_size_input():
    print("\nAvailable board sizes:")
    for i, (size_name, _) in enumerate(BOARD_SIZES.items(), 1):
        print(f"{i}. {size_name}")
    
    while True:
        try:
            choice = int(input("\nSelect board size (1-4): "))
            if 1 <= choice <= 4:
                size_name = list(BOARD_SIZES.keys())[choice - 1]
                return BOARD_SIZES[size_name]
            print("Invalid choice. Please select 1-4.")
        except ValueError:
            print("Please enter a number.")

def get_ai_selection():
    ai_options = ["αβpruning", "αβpruning_improved", "αβ_grok", "MCTS"]
    print("\nAvailable AIs:")
    for i, ai in enumerate(ai_options, 1):
        print(f"{i}. {ai}")
    
    while True:
        try:
            choice1 = int(input("\nSelect AI for Player 1 (1-4): "))
            choice2 = int(input("Select AI for Player 2 (1-4): "))
            if 1 <= choice1 <= 4 and 1 <= choice2 <= 4 and choice1 != choice2:
                return ai_options[choice1 - 1], ai_options[choice2 - 1]
            print("Invalid choices. Please select different AIs (1-4).")
        except ValueError:
            print("Please enter numbers.")

def get_num_games():
    while True:
        try:
            num = int(input("\nEnter number of games to simulate (1-100): "))
            if 1 <= num <= 100:
                return num
            print("Please enter a number between 1 and 100.")
        except ValueError:
            print("Please enter a number.")

def run_simulation(rows, cols, ai1, ai2, num_games):
    results = {
        "total_moves": [],
        "avg_move_times": {ai1: [], ai2: []},
        "chain_moves": {ai1: [], ai2: []}
    }
    
    # Track wins for each AI
    ai_wins = {ai1: 0, ai2: 0}
    draws = 0
    
    for game in range(num_games):
        print(f"\nStarting game {game + 1}/{num_games}")
        sim = FastSimulation(rows, cols)
        
        # Swap AI positions if needed
        if game % 2 == 1:
            ai1, ai2 = ai2, ai1
            
        player1_ai = ai1
        player2_ai = ai2
        
        print(f"Player 1: {player1_ai}")
        print(f"Player 2: {player2_ai}")
        
        # Game loop
        while not is_game_over(sim.lines_drawn, sim.squares):
            current_ai = player1_ai if sim.current_player == 1 else player2_ai
            move = sim.make_move(sim.current_player, current_ai)
            
            if not move:
                break
        
        stats = sim.get_stats()
        print(f"Final scores: {stats['scores']}")
        print(f"Total moves: {stats['total_moves']}")
        print(f"Average move times: {stats['avg_move_time']}")
        print(f"Chain moves: {stats['chain_moves']}")
        print(f"Game phase: {stats['game_phase']}")
        
        # Update results
        p1_score = stats['scores'][1]
        p2_score = stats['scores'][2]
        
        # Track wins by AI name
        if p1_score > p2_score:
            ai_wins[player1_ai] += 1
            print(f"{player1_ai} wins!")
        elif p2_score > p1_score:
            ai_wins[player2_ai] += 1
            print(f"{player2_ai} wins!")
        else:
            draws += 1
            print("It's a tie!")
            
        results["total_moves"].append(stats["total_moves"])
        results["avg_move_times"][player1_ai].append(stats["avg_move_time"][1])
        results["avg_move_times"][player2_ai].append(stats["avg_move_time"][2])
        results["chain_moves"][player1_ai].append(stats["chain_moves"][1])
        results["chain_moves"][player2_ai].append(stats["chain_moves"][2])
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Total games: {num_games}")
    print(f"{ai1} wins: {ai_wins[ai1]}")
    print(f"{ai2} wins: {ai_wins[ai2]}")
    print(f"Draws: {draws}")
    print(f"\nAverage moves per game: {sum(results['total_moves']) / num_games:.1f}")
    print(f"\nAverage move times:")
    print(f"{ai1}: {sum(results['avg_move_times'][ai1]) / num_games:.2f}s")
    print(f"{ai2}: {sum(results['avg_move_times'][ai2]) / num_games:.2f}s")
    print(f"\nAverage chain moves:")
    print(f"{ai1}: {sum(results['chain_moves'][ai1]) / num_games:.1f}")
    print(f"{ai2}: {sum(results['chain_moves'][ai2]) / num_games:.1f}")

def main():
    print("Welcome to Fast AI Simulation!")
    
    # Get board size
    rows, cols = get_board_size_input()
    
    # Get AI selection
    ai1, ai2 = get_ai_selection()
    
    # Get number of games
    num_games = get_num_games()
    
    # Run simulation
    run_simulation(rows, cols, ai1, ai2, num_games)

if __name__ == "__main__":
    main() 