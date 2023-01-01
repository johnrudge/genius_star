from itertools import product, permutations
import matplotlib.pyplot as plt
import numpy as np
import random
import exact_cover as ec

# Unit vectors for the triangular grid
a = np.array([0.0, -1.0])
b = np.array([-np.sqrt(3.0) / 2.0, 0.5])
c = np.array([np.sqrt(3.0) / 2.0, 0.5])


def triangle_coords(t):
    """Given a triangular coordinate triplet t, return the triangle vertices"""
    centre = a * t[0] + b * t[1] + c * t[2]
    if sum(t) % 2 == 1:
        aa = centre + a
        bb = centre + b
        cc = centre + c
        coords = np.vstack([aa, bb, cc, aa])
    if sum(t) % 2 == 0:
        aa = centre - a
        bb = centre - b
        cc = centre - c
        coords = np.vstack([aa, bb, cc, aa])
    return coords


def plot_triangle(t):
    """Plot a triangle given the triangular triplet t"""
    coords = triangle_coords(t)
    plt.plot(coords[:, 0], coords[:, 1], "k")


def plot_block(trigs, col="k", symbol=None):
    """Given the list of triangles, plot the block"""
    for t in trigs:
        coords = triangle_coords(t)
        plt.fill(coords[:, 0], coords[:, 1], col)
        if symbol:
            coord = a * t[0] + b * t[1] + c * t[2]
            plt.plot(coord[0], coord[1], marker="*", color="k", markersize=10)


class Board:
    """Star-shaped board of triangles"""

    def __init__(self):
        up_triangles = [s for s in product(range(8), repeat=3) if sum(s) == 10]
        down_triangles = [s for s in product(range(8), repeat=3) if sum(s) == 11]
        triangles = up_triangles + down_triangles

        self.triangles = [
            t
            for t in triangles
            if (t[0] <= 5 and t[1] <= 5 and t[2] <= 5)
            or (t[0] >= 2 and t[1] >= 2 and t[2] >= 2)
        ]
        self.triangles.sort(key=lambda i: (i[0], -i[1], i[2]))

    def plot(self, numbers=True):
        for i, t in enumerate(self.triangles):
            plot_triangle(t)

            coord = a * t[0] + b * t[1] + c * t[2]
            if numbers:
                plt.text(
                    coord[0], coord[1], i + 1, horizontalalignment="center", zorder=-10
                )
        plt.axis("off")
        plt.axis("square")
        plt.gcf().tight_layout()


class Dice:
    """The seven genius star dice"""

    def __init__(self):
        self.dice_numbers = [
            [12, 13, 23, 24, 32, 33, 41, 42],
            [10, 27, 31],
            [2, 4, 7, 8, 9, 11, 16, 17],
            [1, 5, 15, 34, 44, 48],
            [19, 20, 21, 28, 29, 30],
            [25, 26, 36, 37, 38, 40, 45, 47],
            [18, 22, 39],
        ]

    def roll(self):
        """A random roll of the dice"""
        r = [random.sample(n, 1)[0] for n in self.dice_numbers]
        r.sort()
        return r

    def all_rolls(self):
        """Iterable giving all possible rolls"""
        return product(*self.dice_numbers)


class Piece:
    """Pieces described by a collection of triangles and a colour"""

    def __init__(self, trigs, col):
        self.original_triangles = trigs
        self.col = col

        # Look at all rotations / reflections of piece
        # about centre of a triangle
        set1 = []
        for perm in permutations([0, 1, 2]):
            set1.append(
                [(t[perm[0]], t[perm[1]], t[perm[2]]) for t in self.original_triangles]
            )

        # Reflections about an edge of a triangle,
        # goes from up-triangles to down-triangles
        set2 = []
        for trigs in set1:
            set2.append([(1 - t[0], -t[2], -t[1]) for t in trigs])

        all_triangles = set1 + set2

        # After reflections/ rotations shift origin so first triangle
        # near (0,0,0)
        new_all_triangles = []
        for trigs in all_triangles:
            trigs.sort(key=lambda i: (i[0], -i[1], i[2]))
            start = trigs[0]
            if sum(start) == 0:
                shift = start
            else:
                shift = (start[0] - 1, start[1], start[2])
            new_all_triangles.append(
                [(t[0] - shift[0], t[1] - shift[1], t[2] - shift[2]) for t in trigs]
            )

        # Use set operation to remove duplicates
        # all_triangles now has all possible reflections/ rotations of piece
        self.all_triangles = [tuple(x) for x in new_all_triangles]
        self.all_triangles = list(set(self.all_triangles))

    def triangles(self, perm_idx, shift=(0, 0, 0)):
        """Triangles of the piece after rotation/reflection plus translation"""
        trigs = self.all_triangles[perm_idx]
        return [(t[0] + shift[0], t[1] + shift[1], t[2] + shift[2]) for t in trigs]

    def n_perms(self):
        """Number of unique rotation/reflection operations on piece"""
        return len(self.all_triangles)

    def plot(self, perm_idx=0, shift=(0, 0, 0)):
        """Plot a piece for a given orientation and translation"""
        trigs = self.triangles(perm_idx, shift=shift)
        plot_block(trigs, self.col)
        plt.axis("equal")
        plt.axis("off")

    def plot_all(self):
        """Plot piece in all possible orientations"""
        for i in range(self.n_perms()):
            self.plot(i, shift=(6 * i, 0, 0))


class Pieces(list):
    """List of all game pieces"""

    def __init__(self, star=False):
        pieces = [
            [[0, 0, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [2, -1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [2, -1, 0], [2, -2, 0]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [2, -1, 0], [2, -1, -1]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [1, -1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [1, -1, 1], [0, -1, 1]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [1, 0, -1]],
            [[0, 0, 0], [1, 0, 0], [1, -1, 0], [1, 0, -1], [2, -1, 0]],
        ]
        colors = [
            "royalblue",
            "yellow",
            "pink",
            "red",
            "lawngreen",
            "orange",
            "darkgreen",
            "purple",
            "saddlebrown",
        ]

        if star:
            # If solving for the golden star, last piece is a hexagon
            p = [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, -1, 0], [1, -1, 1], [0, -1, 1]]
            pieces.append(p)
            colors.append("skyblue")
        else:
            # Otherwise, two straight pieces of lenth 3
            p = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
            pieces.append(p)
            pieces.append(p)
            colors.append("deepskyblue")
            colors.append("lightskyblue")

        [self.append(Piece(p, c)) for p, c in zip(pieces, colors)]

    def plot(self, all_orientations=False):
        """Plot all the pieces"""
        if all_orientations:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=(12, 2))
        for j, p in enumerate(self):
            if all_orientations:
                for i in range(p.n_perms()):
                    p.plot(i, shift=(6 * i, -4 * j, +4 * j))
            else:
                p.plot(0, shift=(0, -4 * j, +4 * j))
        fig.tight_layout()


class Game:
    """The genius star game"""

    def __init__(self, roll=None, star=True):
        self.board = Board()
        self.original_pieces = Pieces(star=False)
        self.star_pieces = Pieces(star=True)

        # work out the possible translation vectors for putting pieces on the board
        self.shifts = []
        for t in self.board.triangles:
            if sum(t) == 10:
                self.shifts.append(t)
            if sum(t) == 11:
                self.shifts.append((t[0] - 1, t[1], t[2]))
        self.shifts = list(set(self.shifts))

        if roll:
            self.new_roll(roll, star=star)

    def new_roll(self, roll, star=True):
        """Given a roll of the dice, block the appropriate triangles"""
        self.roll = roll
        self.star = star
        if star:
            self.pieces = self.star_pieces
        else:
            self.pieces = self.original_pieces
        self.blocked_triangles = set([self.board.triangles[i - 1] for i in roll])
        self.triangles = set(self.board.triangles).difference(self.blocked_triangles)
        self.trig_dict = {t: i for i, t in enumerate(self.triangles)}

    def fits(self):
        """Work out where each piece can fit on the board"""
        f = []
        for piece_idx, p in enumerate(self.pieces):
            for perm_idx in range(len(p.all_triangles)):
                for shift in self.shifts:
                    trigs = set(p.triangles(perm_idx, shift))
                    d = trigs.difference(self.triangles)
                    if len(d) == 0:
                        trig_idxs = [self.trig_dict[t] for t in trigs]
                        f.append((piece_idx, trig_idxs, perm_idx, trigs))
        return f

    def matrix(self, f):
        """The incidence matrix corresponding to the fits f"""
        nrows = len(f)
        ncols = len(self.triangles) + len(self.pieces)
        m = np.zeros((nrows, ncols), dtype="int32")
        for i, x in enumerate(f):
            # here j refers to the piece
            j = len(self.triangles) + x[0]
            m[i, j] = 1
            for j in x[1]:
                # here j refers to the triangle
                m[i, j] = 1
        return m

    def solve(self):
        """Solve game using an exact cover problem solver"""
        f = self.fits()
        m = self.matrix(f)
        try:
            sol = ec.get_exact_cover(m)
        except ec.error.NoSolution:
            if self.star:
                # no star solution, try solving without star
                self.new_roll(self.roll, star=False)
                return self.solve()
            else:
                # no solution
                raise ec.error.NoSolution
            return Solution([], self)
        solution = Solution([f[i] for i in sol], self)

        return solution


class Solution:
    """A solved game"""

    def __init__(self, solution, game):
        self.solution = solution
        self.game = game

    def plot(self, show=True):
        self.game.board.plot(numbers=False)

        for t in self.game.blocked_triangles:
            plot_block([t], "white", symbol=True)

        for s in self.solution:
            piece_idx, trigs = s[0], s[3]
            col = self.game.pieces[piece_idx].col
            plot_block(trigs, col)
        plt.gcf().tight_layout()

        if show:
            plt.show()

    def solved(self):
        return len(self.solution) > 0
