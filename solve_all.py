from genius_star import Dice, Game
import numpy as np


def naive_solution_count():
    dice = Dice()
    game = Game()

    star_games = 0

    for i, roll in enumerate(dice.all_rolls()):
        game.new_roll(roll, star=True)
        solution = game.solve()

        if solution.game.star == True:
            star_games += 1

        print(f" {i+1} games solved ({star_games} star games)", end="\r")

    print(f" {i+1} games solved ({star_games} star games)")


def solution_count():
    dice = Dice()
    game = Game()

    # Only solve problems which are related by symmetry once
    rolls = np.array([game.board.equivalent_roll(r) for r in dice.all_rolls()])
    unique_rolls, counts = np.unique(rolls, axis=0, return_counts=True)

    star_games = 0
    solved_games = 0
    unique_star_games = 0
    unique_solved_games = 0

    for i in range(unique_rolls.shape[0]):
        roll = unique_rolls[i, :]
        game.new_roll(roll, star=True)

        solution = game.solve()
        solved_games += counts[i]
        unique_solved_games += 1

        if solution.game.star == True:
            star_games += counts[i]
            unique_star_games += 1

        print(f" {solved_games} games solved ({star_games} star games)", end="\r")

    print(f" {solved_games} games solved ({star_games} star games)")
    print(
        f" {unique_solved_games} unique games solved ({unique_star_games} unique star games)"
    )


if __name__ == "__main__":
    solution_count()
