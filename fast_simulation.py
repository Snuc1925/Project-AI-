import time
import random
from multiprocessing import Pool, cpu_count
from main import (
    generate_squares,
    check_new_squares,
    extract_edge_set,
    edges_of_square,
)

from alphabetapruning import (
    bot_choose_move as bot_choose_move_alphabetapruning,
    TranspositionTable,
    is_game_over
)
from GA import bot_choose_move as bot_choose_move_GA
from MCTS import bot_choose_move_MCTS
from test_main_hard import bot_choose_move as bot_choose_move_hard
from test_main_easy import bot_choose_move as bot_choose_move_easy
# Board size options - mapping from display size to actual grid size
BOARD_SIZES = {
    "2x2": (3, 3),    # 2x2 grid of squares = 3x3 grid of dots
    "3x3": (4, 4),    # 3x3 grid of squares = 4x4 grid of dots
    "3x4": (4, 5),    # 3x4 grid of squares = 4x5 grid of dots
    "4x4": (5, 5)     # 4x4 grid of squares = 5x5 grid of dots
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
        self.chain_count = {1: 0, 2: 0}
        self.three_edge_squares = {1: 0, 2: 0}

    def _GA_ai_move(self):
        return bot_choose_move_GA(self.lines_drawn, self.squares, self.completed_squares[1], self.completed_squares[2])

    def _MCTS_ai_move(self):
        return bot_choose_move_MCTS(self.lines_drawn, self.squares, self.completed_squares[1], self.completed_squares[2])

    def _intuitive_ai_move(self):
        return bot_choose_move_hard(self.lines_drawn, self.squares, self.completed_squares[1], self.completed_squares[2])

    def _random_ai_move(self):
        return bot_choose_move_easy(self.lines_drawn, self.squares, self.completed_squares[1], self.completed_squares[2])
    def _minimax_ai_move(self):
        # Convert sets to lists for minimax
        list_squares_1 = list(self.completed_squares[1])
        list_squares_2 = list(self.completed_squares[2])
        
        # Initialize transposition table for minimax
        self.transposition_table = TranspositionTable()
        
        return bot_choose_move_alphabetapruning(self.lines_drawn, self.squares, list_squares_1, list_squares_2)

    def make_move(self, player, ai_type):
        start_time = time.time()
        
        if ai_type == "alphabetapruning":
            move = self._alphabetapruning_ai_move()
        elif ai_type == "GA":
            move = self._GA_ai_move()
        elif ai_type == "MCTS":
            move = self._MCTS_ai_move()
        elif ai_type == "Intuitive":
            move = self._intuitive_ai_move()
        elif ai_type == "Random":
            move = self._random_ai_move()
        elif ai_type == "Minimax":
            move = self._minimax_ai_move()
        else:
            raise ValueError(f"Unknown AI type: {ai_type}")
            
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
    ai_options = ["alphabetapruning", "GA", "MCTS", "Intuitive", "Random", "Minimax"]
    print("\nAvailable AIs:")
    for i, ai in enumerate(ai_options, 1):
        print(f"{i}. {ai}")
    
    while True:
        try:
            choice1 = int(input("\nSelect AI for Player 1 (1-6): "))
            choice2 = int(input("Select AI for Player 2 (1-6): "))
            if 1 <= choice1 <= 6 and 1 <= choice2 <= 6 and choice1 != choice2:
                return ai_options[choice1 - 1], ai_options[choice2 - 1]
            print("Invalid choices. Please select different AIs (1-6).")
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

def run_single_game(game_params):
    rows, cols, ai1, ai2, game_num = game_params
    print(f"\nStarting game {game_num}")
    sim = FastSimulation(rows, cols)
    
    # Swap AI positions if needed
    if game_num % 2 == 1:
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
    print(f"Three-edge squares: {stats['three_edge_squares']}")
    
    # Return game results
    return {
        "scores": stats['scores'],
        "total_moves": stats['total_moves'],
        "avg_move_time": stats['avg_move_time'],
        "chain_moves": stats['chain_moves'],
        "player1_ai": player1_ai,
        "player2_ai": player2_ai
    }

def run_simulation(rows, cols, ai1, ai2, num_games):
    results = {
        "total_moves": [],
        "avg_move_times": {ai1: [], ai2: []},
        "chain_moves": {ai1: [], ai2: []}
    }
    
    # Track wins for each AI
    ai_wins = {ai1: 0, ai2: 0}
    draws = 0
    
    # Prepare game parameters for parallel processing
    game_params = [(rows, cols, ai1, ai2, i+1) for i in range(num_games)]
    
    # Use number of CPU cores, but limit to number of games
    num_processes = min(cpu_count(), num_games)
    
    # Run games in parallel
    with Pool(processes=num_processes) as pool:
        game_results = pool.map(run_single_game, game_params)
    
    # Process results
    for result in game_results:
        p1_score = result['scores'][1]
        p2_score = result['scores'][2]
        player1_ai = result['player1_ai']
        player2_ai = result['player2_ai']
        
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
            
        results["total_moves"].append(result["total_moves"])
        results["avg_move_times"][player1_ai].append(result["avg_move_time"][1])
        results["avg_move_times"][player2_ai].append(result["avg_move_time"][2])
        results["chain_moves"][player1_ai].append(result["chain_moves"][1])
        results["chain_moves"][player2_ai].append(result["chain_moves"][2])
    
    # Print final results
    print("\n=== Final Results ===")
    print(f"Board size: {rows - 1}x{cols - 1}")
    print(f"Total games: {num_games}")
    print(f"{ai1} wins: {ai_wins[ai1]}")
    print(f"{ai2} wins: {ai_wins[ai2]}")
    print(f"Draws: {draws}")
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