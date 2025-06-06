import random
import numpy as np
from collections import defaultdict
import json
import os
from αβpruning import (
    bot_choose_move,
    evaluate_position,
    predict_chain_reaction,
    find_forced_moves,
    get_opening_move,
    move_priority,
    get_board_size,
    generate_squares,
    extract_edge_set,
    edges_of_square,
    is_game_over,
    check_new_squares,
    get_valid_moves
)

# Board-specific value ranges and constraints
BOARD_SPECIFIC_RANGES = {
    '3x3': {
        'COMPLETE_SQUARE': (15.0, 25.0),
        'THREE_EDGE': (12.0, 18.0),
        'TWO_EDGE': (3.0, 5.0),
        'ONE_EDGE': (0.8, 1.2),
        'CHAIN_PENALTY': (12.0, 18.0),
        'CHAIN_DISRUPTION': (4.0, 8.0),
        'DOUBLE_CROSS': (20.0, 30.0),
        'CHAIN_LENGTH': (1.2, 1.8),
        'CHAIN_CREATION': (8.0, 12.0)
    },
    '4x4': {
        'COMPLETE_SQUARE': (12.0, 18.0),
        'THREE_EDGE': (10.0, 14.0),
        'TWO_EDGE': (2.5, 4.5),
        'ONE_EDGE': (0.8, 1.2),
        'CHAIN_PENALTY': (15.0, 25.0),
        'CHAIN_DISRUPTION': (3.5, 6.5),
        'DOUBLE_CROSS': (15.0, 25.0),
        'CHAIN_LENGTH': (1.4, 1.8),
        'CHAIN_CREATION': (6.0, 10.0)
    },
    '4x5': {
        'COMPLETE_SQUARE': (10.0, 15.0),
        'THREE_EDGE': (8.0, 12.0),
        'TWO_EDGE': (2.0, 4.0),
        'ONE_EDGE': (0.8, 1.2),
        'CHAIN_PENALTY': (18.0, 28.0),
        'CHAIN_DISRUPTION': (3.0, 6.0),
        'DOUBLE_CROSS': (12.0, 22.0),
        'CHAIN_LENGTH': (1.5, 1.9),
        'CHAIN_CREATION': (5.0, 8.0)
    },
    '5x5': {
        'COMPLETE_SQUARE': (8.0, 12.0),
        'THREE_EDGE': (6.0, 10.0),
        'TWO_EDGE': (1.5, 3.5),
        'ONE_EDGE': (0.8, 1.2),
        'CHAIN_PENALTY': (20.0, 30.0),
        'CHAIN_DISRUPTION': (2.5, 5.5),
        'DOUBLE_CROSS': (10.0, 20.0),
        'CHAIN_LENGTH': (1.6, 2.0),
        'CHAIN_CREATION': (4.0, 7.0)
    }
}

class HierarchicalAITrainer:
    def __init__(self, board_size=(3, 3), population_size=30, generations=75):
        self.board_size = board_size
        self.population_size = population_size
        self.generations = generations
        self.squares = generate_squares(board_size[0], board_size[1])
        self.best_values = None
        self.best_score = float('-inf')
        self.best_not_lose_rate = float('-inf')
        self.best_phase = None
        self.board_key = f"{board_size[0]}x{board_size[1]}"
        
        # Calculate board complexity factor
        board_area = board_size[0] * board_size[1]
        if board_area <= 9:  # 3x3
            quality_factor = 1.0
        elif board_area <= 16:  # 4x4
            quality_factor = 0.8
        elif board_area <= 20:  # 4x5
            quality_factor = 0.6
        else:  # 5x5
            quality_factor = 0.4
        
        # Training phases with strength weights and quality thresholds
        self.phases = [
            {
                'name': 'Baseline',
                'ranges': BOARD_SPECIFIC_RANGES[self.board_key],
                'generations': generations // 3,
                'strength_weight': 1.0,
                'initial_quality': -5.5 * quality_factor,  # Even more lenient
                'final_quality': -5.0 * quality_factor,    # Slightly stricter
                'min_games': 8
            },
            {
                'name': 'Exploration',
                'ranges': None,
                'generations': generations // 3,
                'strength_weight': 1.5,
                'initial_quality': -5.0 * quality_factor,  # Start from baseline's end
                'final_quality': -4.5 * quality_factor,    # Slightly stricter
                'min_games': 10
            },
            {
                'name': 'Fine-tuning',
                'ranges': None,
                'generations': generations // 3,
                'strength_weight': 2.0,
                'initial_quality': -4.5 * quality_factor,  # Start from exploration's end
                'final_quality': -4.0 * quality_factor,    # End with moderate standards
                'min_games': 12
            }
        ]
        
        # Initialize population with random variations of current values
        self.population = self._initialize_population()
        
    def _initialize_population(self):
        # Base values from current implementation
        base_values = {
            'COMPLETE_SQUARE': 20.0,
            'THREE_EDGE': 15.0,
            'TWO_EDGE': 4.0,
            'ONE_EDGE': 1.0,
            'CHAIN_PENALTY': 15.0,
            'CHAIN_DISRUPTION': 6.0,
            'DOUBLE_CROSS': 25.0,
            'CHAIN_LENGTH': 1.5,
            'CHAIN_CREATION': 10.0
        }
        
        population = []
        for _ in range(self.population_size):
            # Create variations of base values within current phase ranges
            values = {}
            for key, value in base_values.items():
                min_val, max_val = self.phases[0]['ranges'][key]
                # Random value within the current phase range
                values[key] = round(random.uniform(min_val, max_val), 1)
            population.append(values)
        return population
    
    def _adjust_ranges_for_phase(self, phase, best_values):
        if phase['name'] == 'Exploration':
            # Expand ranges more aggressively around best values
            new_ranges = {}
            for key, value in best_values.items():
                current_range = BOARD_SPECIFIC_RANGES[self.board_key][key]
                range_width = current_range[1] - current_range[0]
                # Allow up to 50% expansion beyond original ranges
                new_ranges[key] = (
                    max(current_range[0] * 0.75, value - range_width * 1.5),
                    min(current_range[1] * 1.25, value + range_width * 1.5)
                )
            return new_ranges
        elif phase['name'] == 'Fine-tuning':
            # Use wider ranges for fine-tuning
            new_ranges = {}
            for key, value in best_values.items():
                current_range = BOARD_SPECIFIC_RANGES[self.board_key][key]
                range_width = (current_range[1] - current_range[0]) * 0.5  # Increased from 0.3
                new_ranges[key] = (
                    max(current_range[0], value - range_width),
                    min(current_range[1], value + range_width)
                )
            return new_ranges
        return phase['ranges']
    
    def _validate_values(self, values, phase_ranges):
        """Validate that values are within acceptable ranges for the current phase"""
        for key, value in values.items():
            min_val, max_val = phase_ranges[key]
            if not (min_val <= value <= max_val):
                return False
        return True
    
    def _adjust_values_to_range(self, values, phase_ranges):
        """Adjust values to be within phase-specific ranges"""
        adjusted = values.copy()
        for key, value in adjusted.items():
            min_val, max_val = phase_ranges[key]
            adjusted[key] = max(min_val, min(value, max_val))
        return adjusted
    
    def _play_game(self, values1, values2):
        lines_drawn = []
        current_player = 1
        scores = {1: 0, 2: 0}
        
        while not is_game_over(lines_drawn, self.squares):
            # Use different evaluation values for each player
            values = values1 if current_player == 1 else values2
            
            # Get move using current evaluation values
            move = self._get_move_with_values(lines_drawn, values)
            if not move:
                break
                
            # Make move
            id1, id2 = move
            new_squares = check_new_squares((id1, id2), lines_drawn, self.squares)
            lines_drawn.append({"id1": id1, "id2": id2, "player": current_player})
            
            if new_squares:
                scores[current_player] += len(new_squares)
            else:
                current_player = 3 - current_player
                
        return scores
    
    def _get_move_with_values(self, lines_drawn, values):
        valid_moves = get_valid_moves(lines_drawn, self.squares)
        if not valid_moves:
            return None
            
        # Sort moves by priority using provided values
        valid_moves.sort(key=lambda m: self._move_priority_with_values(m, lines_drawn, values), reverse=True)
        
        best_move = None
        best_score = float('-inf')
        score_threshold = 0.6
        
        for move in valid_moves:
            new_lines = lines_drawn.copy()
            new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
            
            # Evaluate move using provided values
            score = self._evaluate_move_with_values(move, new_lines, values)
            
            if score > best_score + score_threshold:
                best_score = score
                best_move = move
                
        return best_move if best_move else random.choice(valid_moves)
    
    def _move_priority_with_values(self, move, lines_drawn, values):
        new_lines = lines_drawn.copy()
        new_lines.append({"id1": move[0], "id2": move[1], "player": 2})
        
        score = len(check_new_squares(move, lines_drawn, self.squares)) * values['COMPLETE_SQUARE']
        current_edges = extract_edge_set(new_lines)
        
        # Count three-edge squares
        three_edge_count = 0
        for square in self.squares:
            edges = edges_of_square(square)
            existing = [e for e in edges if e in current_edges]
            if len(existing) == 3:
                three_edge_count += 1
        score += three_edge_count * values['THREE_EDGE']
        
        return score
    
    def _evaluate_move_with_values(self, move, lines_drawn, values):
        score = 0
        current_edges = extract_edge_set(lines_drawn)
        
        # Evaluate completed squares
        completed_squares = len(check_new_squares(move, lines_drawn, self.squares))
        score += completed_squares * values['COMPLETE_SQUARE']
        
        # Evaluate three-edge squares
        three_edge_count = 0
        for square in self.squares:
            edges = edges_of_square(square)
            existing = [e for e in edges if e in current_edges]
            if len(existing) == 3:
                three_edge_count += 1
        score += three_edge_count * values['THREE_EDGE']
        
        # Evaluate chain potential
        chain_score = self._evaluate_chain_potential(move, lines_drawn, values)
        score += chain_score
        
        return score
    
    def _evaluate_chain_potential(self, move, lines_drawn, values):
        score = 0
        current_edges = extract_edge_set(lines_drawn)
        
        # Check for chain creation
        for square in self.squares:
            edges = edges_of_square(square)
            existing = [e for e in edges if e in current_edges]
            if len(existing) == 3:
                missing_edge = [e for e in edges if e not in current_edges][0]
                if missing_edge not in current_edges:
                    score += values['CHAIN_CREATION']
                    
        # Check for chain disruption
        for square in self.squares:
            edges = edges_of_square(square)
            existing = [e for e in edges if e in current_edges]
            if len(existing) == 2 and move in edges:
                score += values['CHAIN_DISRUPTION']
                
        return score
    
    def _tournament_select(self, fitness_scores, tournament_size=5):
        # Ensure tournament size is not larger than available population
        tournament_size = min(tournament_size, len(fitness_scores))
        # If we have very few individuals, use all of them
        if tournament_size <= 1:
            return fitness_scores[0][1]
        tournament = random.sample(fitness_scores, tournament_size)
        return max(tournament, key=lambda x: x[0])[1]
    
    def _crossover(self, parent1, parent2):
        child = {}
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child
    
    def _mutate(self, values, phase_ranges, mutation_rate=0.2):
        mutated = values.copy()
        for key in mutated:
            if random.random() < mutation_rate:
                min_val, max_val = phase_ranges[key]
                # Random value within the current phase range
                mutated[key] = round(random.uniform(min_val, max_val), 1)
        return mutated
    
    def _create_next_generation(self, fitness_scores, phase_ranges):
        new_population = []
        
        # Keep top 10% unchanged (reduced from 20%)
        elite_count = max(1, int(self.population_size * 0.1))
        new_population.extend([values for _, values, _, _, _, _ in fitness_scores[:elite_count]])
        
        # Create rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            # Adjust tournament size based on available population
            tournament_size = min(5, len(fitness_scores))
            parent1 = self._tournament_select(fitness_scores, tournament_size)
            parent2 = self._tournament_select(fitness_scores, tournament_size)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation with higher rate
            child = self._mutate(child, phase_ranges, mutation_rate=0.2)
            
            # Validate and adjust if necessary
            if not self._validate_values(child, phase_ranges):
                child = self._adjust_values_to_range(child, phase_ranges)
            
            new_population.append(child)
        
        return new_population
    
    def _save_best_values(self, phase_name):
        # Create timestamp for this training run
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base training results directory if it doesn't exist
        if not os.path.exists('training_results'):
            os.makedirs('training_results')
            
        # Create timestamped directory for this run
        run_dir = f'training_results/run_{timestamp}'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            
        filename = f'{run_dir}/best_values_{self.board_key}_{phase_name}.json'
        with open(filename, 'w') as f:
            json.dump({
                'board_size': self.board_size,
                'values': self.best_values,
                'not_lose_rate': self.best_not_lose_rate,
                'fitness': self.best_score,
                'strategy_quality': self._evaluate_strategy_quality(self.best_values, 0, 0, 1),  # Add quality
                'phase': phase_name,
                'generation': self.generations,
                'population_size': self.population_size,
                'timestamp': timestamp
            }, f, indent=4)
            
        # Also save a summary of all phases in this run
        summary_file = f'{run_dir}/training_summary.json'
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'board_size': self.board_size,
                'population_size': self.population_size,
                'generations': self.generations,
                'phases': {}
            }
            
        summary['phases'][phase_name] = {
            'best_not_lose_rate': self.best_not_lose_rate,
            'best_fitness': self.best_score,
            'best_values': self.best_values,
            'strategy_quality': self._evaluate_strategy_quality(self.best_values, 0, 0, 1)  # Add quality
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
    
    def _evaluate_strategy_quality(self, values, wins, draws, total_games):
        """Evaluate the quality of a strategy beyond just not-lose rate"""
        win_rate = wins / total_games if total_games > 0 else 0
        draw_rate = draws / total_games if total_games > 0 else 0
        
        # Calculate parameter balance
        parameter_variance = np.var(list(values.values()))
        
        # In optimal play, draws are more common and often the correct outcome
        # We still value wins slightly more to encourage finding winning opportunities
        # but not at the expense of optimal play
        quality_score = (
            win_rate * 0.48 +     
            draw_rate * 0.52 -    # Increased from 0.4
            parameter_variance * 0.1  # Keep parameter balance penalty
        )
        
        return quality_score

    def _calculate_current_threshold(self, phase, generation):
        """Calculate the current quality threshold based on generation progress"""
        progress = generation / phase['generations']
        # Linear interpolation from initial to final threshold
        return phase['initial_quality'] + (phase['final_quality'] - phase['initial_quality']) * progress
    
    def train(self):
        current_phase = 0
        total_generations = 0
        
        while current_phase < len(self.phases):
            phase = self.phases[current_phase]
            print(f"\nStarting {phase['name']} phase")
            print(f"Phase requirements: Initial Quality = {phase['initial_quality']:.2f}, "
                  f"Final Quality = {phase['final_quality']:.2f}, Min Games = {phase['min_games']}")
            
            if phase['ranges'] is None and self.best_values is not None:
                phase['ranges'] = self._adjust_ranges_for_phase(phase, self.best_values)
                print(f"Adjusted ranges for {phase['name']} phase:")
                for key, (min_val, max_val) in phase['ranges'].items():
                    print(f"{key}: ({min_val}, {max_val})")
            
            # Train for this phase
            for generation in range(phase['generations']):
                total_generations += 1
                print(f"\nGeneration {total_generations}/{self.generations}")
                
                # Calculate current quality threshold
                current_quality_threshold = self._calculate_current_threshold(phase, generation)
                print(f"Current quality threshold: {current_quality_threshold:.2f}")
                
                # Evaluate each individual
                fitness_scores = []
                for i, values in enumerate(self.population):
                    # Play against other individuals
                    wins = 0
                    draws = 0
                    total_games = 0
                    
                    for j, opponent_values in enumerate(self.population):
                        if i != j:
                            # Play two games with swapped colors
                            scores1 = self._play_game(values, opponent_values)
                            scores2 = self._play_game(opponent_values, values)
                            
                            # First game
                            if scores1[1] > scores1[2]:
                                wins += 1
                            elif scores1[1] == scores1[2]:
                                draws += 1
                                
                            # Second game
                            if scores2[2] > scores2[1]:
                                wins += 1
                            elif scores2[2] == scores2[1]:
                                draws += 1
                                
                            total_games += 2
                    
                    # Calculate various metrics
                    fitness = (wins + 0.5 * draws) / total_games if total_games > 0 else 0
                    not_lose_rate = (wins + draws) / total_games if total_games > 0 else 0
                    weighted_not_lose_rate = not_lose_rate * phase['strength_weight']
                    strategy_quality = self._evaluate_strategy_quality(values, wins, draws, total_games)
                    
                    # Store all metrics
                    fitness_scores.append((
                        fitness,
                        values,
                        not_lose_rate,
                        weighted_not_lose_rate,
                        strategy_quality,
                        total_games
                    ))
                    
                    print(f"Individual {i + 1}: Win rate = {wins/total_games:.2%}, Draw rate = {draws/total_games:.2%}, "
                          f"Not-lose rate = {not_lose_rate:.2%}, Weighted rate = {weighted_not_lose_rate:.2%}, "
                          f"Quality = {strategy_quality:.3f}, Fitness = {fitness:.2%}, Games = {total_games}")
                
                # Filter and sort individuals using current phase requirements
                valid_scores = [
                    score for score in fitness_scores 
                    if score[5] >= phase['min_games'] and score[4] >= current_quality_threshold
                ]
                
                if not valid_scores:
                    print(f"Warning: No individuals met current quality threshold {current_quality_threshold:.2f}. "
                          f"Using top 30% of individuals by quality.")
                    # Sort by quality and take top 30%
                    fitness_scores.sort(key=lambda x: x[4], reverse=True)
                    valid_scores = fitness_scores[:max(1, len(fitness_scores)//3)]
                
                # Sort by weighted not-lose rate first, then by strategy quality
                valid_scores.sort(key=lambda x: (x[3], x[4]), reverse=True)
                
                # Update best values if weighted score is better or equal but quality is better
                current_best = (self.best_score, self.best_values, self.best_not_lose_rate, self.best_phase)
                new_best = (valid_scores[0][3], valid_scores[0][1], valid_scores[0][2], phase['name'])
                
                if (new_best[0] > current_best[0] or 
                    (new_best[0] == current_best[0] and valid_scores[0][4] > self._evaluate_strategy_quality(current_best[1], 0, 0, 1))):
                    self.best_score = new_best[0]
                    self.best_values = new_best[1]
                    self.best_not_lose_rate = new_best[2]
                    self.best_phase = new_best[3]
                    print(f"\nNew best values found in {phase['name']} phase!")
                    print(f"Not-lose rate: {self.best_not_lose_rate:.2%}")
                    print(f"Weighted rate: {self.best_score:.2%}")
                    print(f"Strategy quality: {valid_scores[0][4]:.3f}")
                    print("Values:", self.best_values)
                
                # Create next generation
                self.population = self._create_next_generation(valid_scores, phase['ranges'])
            
            # Save best values at the end of each phase
            self._save_best_values(phase['name'])
            current_phase += 1

def main():
    # Train for different board sizes
    board_sizes = [(3, 3), (4, 4), (4, 5), (5, 5)]
    
    for board_size in board_sizes:
        print(f"\nTraining for board size {board_size[0]}x{board_size[1]}")
        trainer = HierarchicalAITrainer(board_size=board_size)
        trainer.train()

if __name__ == "__main__":
    main() 