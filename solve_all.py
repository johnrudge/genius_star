from genius_star import Dice, Game

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
