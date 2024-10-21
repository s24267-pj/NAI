from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class OddOrEven(TwoPlayerGame):
    def __init__(self, players):
        """
        Initializes the Odd or Even game.

        :param players: List of players (human and AI).
        """
        self.players = players
        self.current_player = 1
        self.numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.player1_choices = []
        self.player2_choices = []
        self.current_round_choices = []
        self.player1_score = 0
        self.player2_score = 0

    def possible_moves(self):
        """
        Returns a list of available moves (numbers) that have not been chosen by either player.

        :return: List of available moves.
        """
        return [str(n) for n in self.numbers if n not in self.player1_choices and n not in self.player2_choices]

    def make_move(self, move):
        """
        Executes a move, updating the players choices and the current round choices.

        :param move: The number chosen by the current player.
        """
        move = int(move)
        self.current_round_choices.append(move)

        if self.current_player == 1:
            self.player1_choices.append(move)
        else:
            self.player2_choices.append(move)

        if len(self.current_round_choices) == 2:
            self.update_scores()
            self.current_round_choices = []

    def update_scores(self):
        """
        Updates the players scores based on the sum of choices in the current round.
        Player 1 (Odd) scores a point if the sum is odd, while Player 2 (Even) scores a point if the sum is even.
        """
        total_sum = sum(self.current_round_choices)
        if total_sum % 2 == 0:
            self.player2_score += 1
        else:
            self.player1_score += 1

    def is_over(self):
        """
        Checks if the game is over (all numbers have been chosen).

        :return: True if the game is over, otherwise False.
        """
        return len(self.player1_choices) + len(self.player2_choices) == len(self.numbers)

    def show(self):
        """
        Displays the players choices and the current score.
        """
        print(f"Player 1 choices: {self.player1_choices}")
        print(f"Player 2 choices: {self.player2_choices}")
        print(f"Current score - Player 1 (Odd): {self.player1_score}, Player 2 (Even): {self.player2_score}")

    def scoring(self):
        """
        Returns the evaluation of the game state, which is the difference in scores between the AI player and the human player, multiplied by 10.

        :return: Score difference multiplied by 10.
        """
        return (self.player2_score - self.player1_score) * 10

    def ai_move(self):
        """
        Executes the AI move by analyzing possible moves and selecting the best one based on evaluation.
        """
        possible_moves = self.possible_moves()
        best_move = None
        best_score = float('-inf')

        for move in possible_moves:
            self.make_move(move)
            score = self.scoring()

            if self.current_player == 2:
                self.player2_choices.pop()
            else:
                self.player1_choices.pop()

            if self.current_round_choices:
                self.current_round_choices.pop()

            if score > best_score:
                best_score = score
                best_move = move

        self.make_move(best_move)

    def play(self):
        """
        Starts the game, executing players moves until the game is over.
        """
        while not self.is_over():
            self.show()
            if self.current_player == 1:
                move = input(f"Player 1, choose a number from available moves: {self.possible_moves()}: ")
                self.make_move(move)
            else:
                print("Player 2 (AI) is making a move...")
                self.ai_move()
            self.current_player = 2 if self.current_player == 1 else 1


# AI setup
ai_algo = Negamax(10)

# Start the game
game = OddOrEven([Human_Player(), AI_Player(ai_algo)])
game.play()
