# Connect 4 AI Game

A Python implementation of **Connect 4** with AI opponents using three different search algorithms: **Minimax**, **Alpha-Beta Pruning**, and **Expectiminimax**.

## Features

### Three AI Algorithms
- **Minimax**: Classic adversarial search
- **Alpha-Beta**: Optimized Minimax with pruning
- **Expectiminimax**: Accounts for physical uncertainty in disc placement

### Customizable Game Settings
- Adjustable board dimensions (*rows × columns*)
- Configurable search depth (*K parameter*)
- Real-time algorithm switching

### Visual Interface
- Clean GUI built with *Tkinter*
- Color-coded pieces (*Red = Human, Yellow = AI*)
- Real-time game status and scoring

## Requirements
- Python 3.x
- tkinter (usually included with Python)

## Game Setup
1. *Select Algorithm*: Choose from Minimax, Alpha-Beta, or Expectiminimax
2. *Set Board Size*: Minimum 6 rows × 7 columns
3. *Choose Depth: Set search depth (K*) for AI decision making
4. *Click Start* to begin playing

## Game Rules
- Human plays as Red, AI as *Yellow*
- Take turns dropping discs into columns
- The player who gets more *4s in a row* (horizontally, vertically, or diagonally) wins
- Game ends when the board is full
- *Human always starts first*

## Algorithm Details

### Heuristic Evaluation
The AI evaluates board states considering:
- Connected pieces
- Potential threats
- Center column control
- Blocking opponent opportunities

### Expectimax Feature
Models physical uncertainty where discs may slide to adjacent columns:
- *60% chance* disc lands in chosen column
- *20% chance* slides left to C-1 (if valid)
- *20% chance* slides right to C+1 (if valid)
- *40% chance* if only one side available 

### Performance Optimization
- Memoization for repeated board states
- Center-column move ordering for better AI decision-making
- Alpha-beta pruning for efficient search

## Controls
- Click the *"↓" buttons* above each column to make your move
- The AI automatically responds after each human move

## Enjoy!
Play against intelligent AI opponents with different strategic approaches and see how each algorithm makes its moves. Experiment with board sizes and search depths to challenge yourself!
