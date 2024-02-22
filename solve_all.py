from itertools import combinations
import numpy as np
from genius_star import Dice, Game, NoSolution


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


def count_dice_roll_solutions():
    dice = Dice()
    solution_count(dice.all_rolls())


def count_all_solutions():
    rolls = combinations(range(1, 49), 7)
    solution_count(rolls)


def solution_count(rolls):
    game = Game()

    star_games = 0
    solved_games = 0
    nosolution_games = 0
    unique_star_games = 0
    unique_solved_games = 0
    unique_nosolution_games = 0

    star_seen = set()
    nostar_seen = set()
    nosolution_seen = set()

    for r in rolls:
        roll = tuple(game.board.equivalent_roll(r))

        # Only solve problems which are related by symmetry once
        if roll in star_seen:
            star_games += 1
            solved_games += 1
            continue
        if roll in nostar_seen:
            solved_games += 1
            continue
        if roll in nosolution_seen:
            nosolution_games += 1
            continue

        game.new_roll(roll, star=True)
        try:
            solution = game.solve()
        except NoSolution:
            unique_nosolution_games += 1
            nosolution_games += 1
            nosolution_seen.add(roll)
            continue

        solved_games += 1
        unique_solved_games += 1

        if solution.game.star == True:
            star_games += 1
            unique_star_games += 1
            star_seen.add(roll)
        else:
            nostar_seen.add(roll)

        print(
            f" {unique_solved_games} unique games solved ({unique_star_games} unique star games)",
            end="\r",
        )

    print(
        f" {unique_solved_games} unique games solved ({unique_star_games} unique star games)"
    )
    print(f" {solved_games} games solved ({star_games} star games)")

    if nosolution_games > 0:
        print(
            f"Searched {solved_games + nosolution_games} games ({unique_solved_games+unique_nosolution_games} unique)"
        )


if __name__ == "__main__":
    print("Counting dice roll solutions:")
    count_dice_roll_solutions()
    print("Counting solutions for all blocker positions (takes a long time!):")
    count_all_solutions()
