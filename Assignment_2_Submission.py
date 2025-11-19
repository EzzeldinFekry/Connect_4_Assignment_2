import tkinter as tk
from tkinter import messagebox, ttk
import copy
import math
import time

class Connect4:
    """Connect 4 game with AI using Minimax, Alpha-Beta, and Expectimax algorithms"""
    
    def __init__(self):
        # Initialize GUI window
        self.root = tk.Tk()
        self.root.title("Connect 4 AI")
        self.root.geometry("800x650")
        
        # Game state variables
        self.board = None  # 2D list representing game board
        self.rows = self.cols = self.depth = 0  # Board dimensions and search depth
        self.algo = "alpha_beta"  # Selected algorithm
        self.player = 1  # Current player: 1=Human(Red), 2=AI(Yellow)
        
        # Performance tracking
        self.nodes_expanded = 0
        self.start_time = 0
        
        self.setup_ui()
    
    # ==================== GUI SETUP ====================
    def setup_ui(self):
        """Create all GUI components"""
        # Configuration panel at top
        config_frame = tk.Frame(self.root, bg="#2c3e50", padx=10, pady=10)
        config_frame.pack(fill=tk.X)
        
        # Algorithm selection dropdown
        tk.Label(config_frame, text="Algorithm:", bg="#2c3e50", fg="white").grid(row=0, column=0, padx=5)
        self.algo_var = tk.StringVar(value="alpha_beta")
        ttk.Combobox(config_frame, textvariable=self.algo_var, 
                     values=["minimax", "alpha_beta", "expectimax"], 
                     state="readonly", width=12).grid(row=0, column=1, padx=5)
        
        # Board dimensions inputs
        tk.Label(config_frame, text="Rows:", bg="#2c3e50", fg="white").grid(row=0, column=2, padx=5)
        self.rows_var = tk.StringVar(value="6")
        tk.Entry(config_frame, textvariable=self.rows_var, width=5).grid(row=0, column=3, padx=5)
        
        tk.Label(config_frame, text="Cols:", bg="#2c3e50", fg="white").grid(row=0, column=4, padx=5)
        self.cols_var = tk.StringVar(value="7")
        tk.Entry(config_frame, textvariable=self.cols_var, width=5).grid(row=0, column=5, padx=5)
        
        # Search depth (K parameter)
        tk.Label(config_frame, text="Depth(K):", bg="#2c3e50", fg="white").grid(row=0, column=6, padx=5)
        self.depth_var = tk.StringVar(value="4")
        tk.Entry(config_frame, textvariable=self.depth_var, width=5).grid(row=0, column=7, padx=5)
        
        tk.Button(config_frame, text="Start", command=self.start_game, 
                 bg="#27ae60", fg="white").grid(row=0, column=8, padx=10)
        
        # Status and score labels
        self.info = tk.Label(self.root, text="Configure and start game", 
                            bg="#34495e", fg="white", font=("Arial", 11))
        self.info.pack(fill=tk.X, pady=5)
        
        self.score = tk.Label(self.root, text="Human: 0 | AI: 0", bg="#34495e", fg="white")
        self.score.pack(fill=tk.X)
        
        # Game board display area
        self.game_frame = tk.Frame(self.root, bg="#1a252f")
        self.game_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    
    def create_board(self):
        """Create visual game board with column buttons and canvas"""
        # Clear previous board
        for widget in self.game_frame.winfo_children():
            widget.destroy()
        
        # Column selection buttons
        btn_frame = tk.Frame(self.game_frame, bg="#1a252f")
        btn_frame.pack(pady=10)
        self.buttons = []
        
        for col in range(self.cols):
            btn = tk.Button(btn_frame, text="↓", width=3, 
                          command=lambda c=col: self.human_move(c),
                          bg="#3498db", fg="white", font=("Arial", 11, "bold"))
            btn.pack(side=tk.LEFT, padx=2)
            self.buttons.append(btn)
        
        # Canvas for drawing game pieces
        self.canvas = tk.Canvas(self.game_frame, 
                               width=self.cols*60+20, height=self.rows*60+20,
                               bg="#2980b9", highlightthickness=0)
        self.canvas.pack()
        self.draw_board()
    
    def draw_board(self):
        """Render current board state on canvas"""
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c*60+10, r*60+10
                # Color: white=empty, red=human, yellow=AI
                color = {0: "white", 1: "#e74c3c", 2: "#f39c12"}[self.board[r][c]]
                self.canvas.create_oval(x, y, x+50, y+50, fill=color, outline="#34495e", width=2)
    
    # ==================== GAME FLOW ====================
    def start_game(self):
        """Initialize new game with configured parameters"""
        try:
            # Parse and validate inputs
            self.rows = int(self.rows_var.get())
            self.cols = int(self.cols_var.get())
            self.depth = int(self.depth_var.get())
            self.algo = self.algo_var.get()
            
            if self.rows < 6 or self.cols < 7 or self.depth < 1:
                messagebox.showerror("Error", "Minimum: 6 rows, 7 columns, depth 1")
                return
                
            # Initialize empty board (0=empty, 1=human, 2=AI)
            self.board = [[0]*self.cols for _ in range(self.rows)]
            self.player = 1  # Human starts
            self.create_board()
            self.info.config(text="Your turn (Red)")
            
            print(f"\n{'='*60}\nNEW GAME - {self.algo.upper()}, Depth: {self.depth}\n{'='*60}")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
    
    def human_move(self, col):
        """Handle human player move"""
        # Validate move
        if self.player != 1 or not self.valid_move(col):
            return
        
        # Drop piece in column
        row = self.drop_piece(col, 1)
        self.draw_board()
        
        print(f"\nHuman played column {col}")
        self.print_board()
        
        # Check if board full
        if self.board_full():
            self.end_game()
            return
        
        # Switch to AI turn
        self.player = 2
        self.info.config(text="AI thinking...")
        self.root.update()
        self.root.after(300, self.ai_move)
    
    def ai_move(self):
        """Execute AI move using selected algorithm"""
        print(f"\n{'='*60}\nAI TURN - {self.algo.upper()}\n{'='*60}")
        
        # Reset performance counters
        self.nodes_expanded = 0
        self.start_time = time.time()
        
        # Select and run algorithm
        if self.algo == "minimax":
            col, score = self.minimax_search()
        elif self.algo == "alpha_beta":
            col, score = self.alphabeta_search()
        else:  # expectimax
            col, score = self.expectimax_search()
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Make move
        row = self.drop_piece(col, 2)
        self.draw_board()
        
        print(f"\nAI chose column {col} (score: {score:.2f})")
        print(f"Performance: {self.nodes_expanded} nodes expanded in {elapsed_time:.3f} seconds")
        print(f"Nodes/second: {self.nodes_expanded/elapsed_time:.0f}")
        self.print_board()
        
        # Check if board full
        if self.board_full():
            self.end_game()
            return
        
        # Switch to human turn
        self.player = 1
        self.info.config(text="Your turn (Red)")
    
    def end_game(self):
        """Handle game over - count scores and display winner"""
        human_score = self.count_fours(self.board, 1)
        ai_score = self.count_fours(self.board, 2)
        
        self.score.config(text=f"Human: {human_score} | AI: {ai_score}")
        
        # Determine winner
        if human_score > ai_score:
            result = "You Win! 🎉"
        elif ai_score > human_score:
            result = "AI Wins! 🤖"
        else:
            result = "Tie Game! 🤝"
        
        result += f"\nFinal Score - Human: {human_score} | AI: {ai_score}"
        
        print(f"\n{'='*60}\nGAME OVER\nHuman: {human_score} | AI: {ai_score}\n{'='*60}")
        
        self.info.config(text="Game Over!")
        messagebox.showinfo("Game Over", result)
        
        # Disable column buttons
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)
    
    # ==================== MINIMAX ALGORITHM ====================
    def minimax_search(self):
        """Root call for minimax algorithm - finds best move for AI"""
        print(f"\n{'─'*60}\nMINIMAX TREE\n{'─'*60}")
        best_col, best_score = None, -math.inf
        valid_cols = [c for c in range(self.cols) if self.valid_move(c)]
        
        print(f"Root [MAX, AI] - Valid moves: {valid_cols}, Current H={self.heuristic()}")
        
        # Try each valid column
        for col in valid_cols:
            board_copy = self.make_move(col, 2)
            self.nodes_expanded += 1  # Count root's children
            score = self.minimax(board_copy, self.depth-1, False, 1, col)
            print(f"  → Column {col}: Score = {score}")
            
            if score > best_score:
                best_score = score
                best_col = col
        
        print(f"{'─'*60}\nBEST MOVE: Column {best_col}, Score {best_score}\n{'─'*60}")
        return best_col, best_score
    
    def minimax(self, board, depth, is_max, level, last_move):
        """
        Recursive minimax algorithm
        - depth: remaining search depth
        - is_max: True if maximizing player (AI), False if minimizing (human)
        - level: tree depth for printing
        - last_move: last column played
        """
        indent = "  " * level
        
        # Base case: leaf node (depth=0 or board full)
        if depth == 0 or self.is_full(board):
            h = self.heuristic(board)
            print(f"{indent}[Depth {self.depth-depth}] {'MAX' if is_max else 'MIN'} "
                  f"Move={last_move} → LEAF, Heuristic={h}")
            return h
        
        valid_cols = [c for c in range(self.cols) if self.is_valid(board, c)]
        
        if is_max:  # AI's turn (maximize)
            max_score = -math.inf
            for col in valid_cols:
                new_board = self.apply_move(board, col, 2)
                self.nodes_expanded += 1
                score = self.minimax(new_board, depth-1, False, level+1, col)
                max_score = max(max_score, score)
            
            print(f"{indent}[Depth {self.depth-depth}] MAX returns {max_score}")
            return max_score
            
        else:  # Human's turn (minimize)
            min_score = math.inf
            for col in valid_cols:
                new_board = self.apply_move(board, col, 1)
                self.nodes_expanded += 1
                score = self.minimax(new_board, depth-1, True, level+1, col)
                min_score = min(min_score, score)
            
            print(f"{indent}[Depth {self.depth-depth}] MIN returns {min_score}")
            return min_score
        
    # ==================== ALPHA-BETA PRUNING ====================
    def alphabeta_search(self):
        """Root call for alpha-beta pruning - optimized minimax"""
        print(f"\n{'─'*60}\nALPHA-BETA PRUNING TREE\n{'─'*60}")
        best_col, best_score = None, -math.inf
        alpha, beta = -math.inf, math.inf
        valid_cols = [c for c in range(self.cols) if self.valid_move(c)]
        
        print(f"Root [MAX, AI] - Valid: {valid_cols}, α={alpha}, β={beta}, H={self.heuristic()}")
        
        for col in valid_cols:
            board_copy = self.make_move(col, 2)
            self.nodes_expanded += 1
            score = self.alphabeta(board_copy, self.depth-1, alpha, beta, False, 1, col)
            print(f"  → Column {col}: Score = {score}, α updated to {max(alpha, score)}")
            
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, score)  # Update alpha at root
        
        print(f"{'─'*60}\nBEST MOVE: Column {best_col}, Score {best_score}\n{'─'*60}")
        return best_col, best_score
    
    def alphabeta(self, board, depth, alpha, beta, is_max, level, last_move):
        """
        Alpha-beta pruning algorithm
        - alpha: best score for maximizer
        - beta: best score for minimizer
        Prunes branches that can't affect final decision
        """
        indent = "  " * level
        
        # Base case
        if depth == 0 or self.is_full(board):
            h = self.heuristic(board)
            print(f"{indent}[D{self.depth-depth}] {'MAX' if is_max else 'MIN'} "
                  f"M={last_move} → H={h}, α={alpha:.1f}, β={beta:.1f}")
            return h
        
        valid_cols = [c for c in range(self.cols) if self.is_valid(board, c)]
        
        if is_max:  # AI maximizes
            max_score = -math.inf
            for col in valid_cols:
                new_board = self.apply_move(board, col, 2)
                self.nodes_expanded += 1
                score = self.alphabeta(new_board, depth-1, alpha, beta, False, level+1, col)
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                
                # Beta cutoff: minimizer won't choose this path
                if beta <= alpha:
                    print(f"{indent}✂ PRUNED (β={beta:.1f} ≤ α={alpha:.1f})")
                    break
            
            print(f"{indent}[D{self.depth-depth}] MAX returns {max_score}")
            return max_score
            
        else:  # Human minimizes
            min_score = math.inf
            for col in valid_cols:
                new_board = self.apply_move(board, col, 1)
                self.nodes_expanded += 1
                score = self.alphabeta(new_board, depth-1, alpha, beta, True, level+1, col)
                min_score = min(min_score, score)
                beta = min(beta, score)
                
                # Alpha cutoff: maximizer won't choose this path
                if beta <= alpha:
                    print(f"{indent}✂ PRUNED (β={beta:.1f} ≤ α={alpha:.1f})")
                    break
            
            print(f"{indent}[D{self.depth-depth}] MIN returns {min_score}")
            return min_score
    
    # ==================== EXPECTIMAX ALGORITHM ====================
    def expectimax_search(self):
        """
        Root call for expectimax - handles physical uncertainty in disc placement
        When a player chooses column C:
        - 60% chance disc lands in column C
        - 20% chance disc slides left to C-1 (if valid)
        - 20% chance disc slides right to C+1 (if valid)
        - If only one adjacent valid: 40% to that side, 60% to chosen
        
        OPTIMIZATION: Uses memoization to cache repeated board states
        """
        print(f"\n{'─'*60}\nEXPECTIMAX TREE (OPTIMIZED)\n{'─'*60}")
        print("Disc placement: 60% chosen, 20% left, 20% right (40% if one side blocked)")
        
        best_col, best_score = None, -math.inf
        valid_cols = [c for c in range(self.cols) if self.valid_move(c)]
        
        # Order columns from center outward (better moves usually in center)
        center = self.cols // 2
        valid_cols.sort(key=lambda c: abs(c - center))
        
        print(f"Root [MAX, AI] - Valid (ordered): {valid_cols}, H={self.heuristic()}")
        
        # Clear memoization cache for new search
        self.expectimax_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # AI tries each column and computes expected value
        for col in valid_cols:
            # AI chooses column, then expectation node handles physical uncertainty
            self.nodes_expanded += 1
            expected = self.compute_expected_outcome(self.board, col, 2, self.depth-1, False, 0)
            print(f"  → Column {col}: Expected Value = {expected:.2f}")
            
            if expected > best_score:
                best_score = expected
                best_col = col
        
        total_lookups = self.cache_hits + self.cache_misses
        hit_rate = 100 * self.cache_hits / total_lookups if total_lookups > 0 else 0
        print(f"\nCache: {self.cache_hits}/{total_lookups} hits ({hit_rate:.1f}%)")
        print(f"{'─'*60}\nBEST MOVE: Column {best_col}, Score {best_score:.2f}\n{'─'*60}")
        return best_col, best_score
    
    def compute_expected_outcome(self, board, chosen_col, player, depth, is_max, level):
        """
        Compute expected value when player chooses a column
        Accounts for physical uncertainty in where disc actually lands
        """
        indent = "  " * level
        print(f"{indent}Expectation for col {chosen_col}:")
        
        expected_value = 0.0
        
        # Determine which columns disc might land in
        left_col = chosen_col - 1
        right_col = chosen_col + 1
        
        can_go_left = left_col >= 0 and self.is_valid(board, left_col)
        can_go_right = right_col < self.cols and self.is_valid(board, right_col)
        
        # Calculate probabilities based on available adjacent columns
        if can_go_left and can_go_right:
            # Both sides available: 60% chosen, 20% left, 20% right
            probs = [(chosen_col, 0.6), (left_col, 0.2), (right_col, 0.2)]
        elif can_go_left:
            # Only left available: 60% chosen, 40% left
            probs = [(chosen_col, 0.6), (left_col, 0.4)]
        elif can_go_right:
            # Only right available: 60% chosen, 40% right
            probs = [(chosen_col, 0.6), (right_col, 0.4)]
        else:
            # No adjacent columns: 100% chosen
            probs = [(chosen_col, 1.0)]
        
        # Compute expected value over all possible landing positions
        for landing_col, prob in probs:
            new_board = self.apply_move(board, landing_col, player)
            self.nodes_expanded += 1
            score = self.expectimax(new_board, depth, is_max, level+1, landing_col)
            expected_value += prob * score
            
            print(f"{indent}  → Col {landing_col} (P={prob}): {score:.2f}")
        
        print(f"{indent}  Expected: {expected_value:.2f}")
        return expected_value
    
    def expectimax(self, board, depth, is_max, level, last_move):
        """
        Expectimax algorithm with physical uncertainty and memoization
        - MAX nodes: AI deterministically chooses best column
        - MIN nodes: Human deterministically chooses (treated as expectation over choices)
        - After each choice: EXPECTATION node handles physical uncertainty
        """
        # Check cache first
        cache_key = self.board_to_key(board, depth, is_max)
        if cache_key in self.expectimax_cache:
            if not hasattr(self, 'cache_hits'):
                self.cache_hits = 0
            self.cache_hits += 1
            return self.expectimax_cache[cache_key]
        
        if not hasattr(self, 'cache_misses'):
            self.cache_misses = 0
        self.cache_misses += 1
        
        indent = "  " * level
        
        # Base case: leaf node
        if depth == 0 or self.is_full(board):
            h = self.heuristic(board)
            print(f"{indent}[D{self.depth-depth}] LEAF → H={h}")
            self.expectimax_cache[cache_key] = h
            return h
        
        valid_cols = [c for c in range(self.cols) if self.is_valid(board, c)]
        
        if is_max:  # AI's turn - maximize expected value
            print(f"{indent}[D{self.depth-depth}] MAX node (AI)")
            max_score = -math.inf
            
            # AI chooses column that maximizes expected value
            for col in valid_cols:
                expected = self.compute_expected_outcome(board, col, 2, depth-1, False, level+1)
                max_score = max(max_score, expected)
            
            print(f"{indent}MAX → {max_score:.2f}")
            self.expectimax_cache[cache_key] = max_score
            return max_score
            
        else:  # Human's turn - minimize expected value (opponent model)
            print(f"{indent}[D{self.depth-depth}] MIN node (Human)")
            min_score = math.inf
            
            # Human chooses column that minimizes AI's expected value
            for col in valid_cols:
                expected = self.compute_expected_outcome(board, col, 1, depth-1, True, level+1)
                min_score = min(min_score, expected)
            
            print(f"{indent}MIN → {min_score:.2f}")
            self.expectimax_cache[cache_key] = min_score
            return min_score
    
    def board_to_key(self, board, depth, is_max):
        """Convert board state to hashable key for memoization"""
        # Flatten board and include depth and player type
        return (tuple(tuple(row) for row in board), depth, is_max)
    
    # ==================== HEURISTIC EVALUATION ====================
    def heuristic(self, board=None):
        """
        Evaluate board position
        Returns: positive if AI winning, negative if human winning
        
        Factors considered:
        1. Connected pieces (4-in-row worth most)
        2. Potential winning sequences (3 with 1 empty)
        3. Blocking opponent threats
        4. Center column control (strategic advantage)
        5. Current score (completed 4-in-rows)
        """
        if board is None:
            board = self.board
        
        score = 0
        
        # Evaluate all possible 4-cell windows
        # Horizontal windows
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self.eval_window(window)
        
        # Vertical windows
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = [board[r+i][c] for i in range(4)]
                score += self.eval_window(window)
        
        # Diagonal (top-left to bottom-right)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.eval_window(window)
        
        # Diagonal (bottom-left to top-right)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self.eval_window(window)
        
        # Bonus for center column control (strategic position)
        center_col = self.cols // 2
        center_count = sum(1 for r in range(self.rows) if board[r][center_col] == 2)
        score += center_count * 3
        
        # Add current score differential
        ai_fours = self.count_fours(board, 2)
        human_fours = self.count_fours(board, 1)
        score += (ai_fours - human_fours) * 1000
        
        return score
    
    def eval_window(self, window):
        """
        Evaluate a 4-cell window for strategic value
        Returns positive for AI advantage, negative for human advantage
        """
        ai_count = window.count(2)
        human_count = window.count(1)
        empty_count = window.count(0)
        
        # AI winning patterns
        if ai_count == 4:
            return 100000  # Win
        if ai_count == 3 and empty_count == 1:
            return 100  # Strong threat
        if ai_count == 2 and empty_count == 2:
            return 10  # Potential
        
        # Human threatening patterns (must block)
        if human_count == 4:
            return -100000  # Loss
        if human_count == 3 and empty_count == 1:
            return -150  # Critical threat (higher penalty to prioritize blocking)
        if human_count == 2 and empty_count == 2:
            return -10  # Potential threat
        
        return 0
    
    # ==================== BOARD UTILITY FUNCTIONS ====================
    def valid_move(self, col):
        """Check if column has space for a piece"""
        return self.board[0][col] == 0
    
    def is_valid(self, board, col):
        """Check if column has space in given board state"""
        return board[0][col] == 0
    
    def drop_piece(self, col, player):
        """Drop piece in column, return row where it landed"""
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = player
                return r
        return None
    
    def make_move(self, col, player):
        """Create new board with move applied (for current board)"""
        new_board = copy.deepcopy(self.board)
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board
    
    def apply_move(self, board, col, player):
        """Create new board with move applied (for any board)"""
        new_board = copy.deepcopy(board)
        for r in range(self.rows - 1, -1, -1):
            if new_board[r][col] == 0:
                new_board[r][col] = player
                break
        return new_board
    
    def board_full(self):
        """Check if current board is full"""
        return all(self.board[0][c] != 0 for c in range(self.cols))
    
    def is_full(self, board):
        """Check if given board is full"""
        return all(board[0][c] != 0 for c in range(self.cols))
    
    def count_fours(self, board, player):
        """Count completed 4-in-a-rows for a player"""
        count = 0
        
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(board[r][c+i] == player for i in range(4)):
                    count += 1
        
        # Vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(board[r+i][c] == player for i in range(4)):
                    count += 1
        
        # Diagonal (TL to BR)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(board[r+i][c+i] == player for i in range(4)):
                    count += 1
        
        # Diagonal (BL to TR)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(board[r-i][c+i] == player for i in range(4)):
                    count += 1
        
        return count
    
    def print_board(self):
        """Print board state to console with heuristic value"""
        print("\nCurrent Board State:")
        print("  " + " ".join(str(i) for i in range(self.cols)))
        for row in self.board:
            display = " ".join("." if c==0 else ("R" if c==1 else "Y") for c in row)
            print(f"│ {display} │")
        print(f"Heuristic Evaluation: {self.heuristic()}\n")
    
    # ==================== MAIN ====================
    def run(self):
        """Start the game application"""
        self.root.mainloop()


# Run the game
if __name__ == "__main__":
    Connect4().run()
