from itertools import product
import random
import matplotlib.pyplot as plt
import numpy as np
from xcover import covers_bool

# Unit vectors for the triangular grid
a = np.array([0.0, -1.0])
b = np.array([-np.sqrt(3.0) / 2.0, 0.5])
c = np.array([np.sqrt(3.0) / 2.0, 0.5])


def triangle_coords(t):
    """Given a triangular coordinate triplet t, return the Cartesian vertices"""
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
        self.trig_dict = {t: i for i, t in enumerate(self.triangles)}
        self.point_group = PointGroup(shift=(3, 4, 3))
        self.calculate_equivalence()
        self.calculate_shifts()

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
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.xlim((-5.25, 5.25))

    def calculate_shifts(self):
        """Work out the possible translation vectors for putting pieces on the board"""
        shifts = []
        for t in self.triangles:
            if sum(t) == 10:
                shifts.append(t)
            if sum(t) == 11:
                shifts.append((t[0] - 1, t[1], t[2]))
        self.shifts = list(set(shifts))

    def intersection(self, piece, perm_idx, shift, piece_idx=None):
        """Work out how an individual piece intersects the board"""
        trigs = set(piece.triangles(perm_idx, shift))
        d = trigs.difference(self.triangles)
        if len(d) == 0:
            trig_idxs = [self.trig_dict[t] for t in trigs]
            return (piece_idx, trig_idxs)
        return None

    def fits(self, pieces):
        """Work out where all pieces can fit on the board"""
        fits = [
            self.intersection(piece, perm_idx, shift, piece_idx)
            for piece_idx, piece in enumerate(pieces)
            for perm_idx in range(piece.n_perms())
            for shift in self.shifts
        ]
        return [intersect for intersect in fits if intersect]

    def incidence_matrix(self, fits):
        """Convert fits information to a boolean incidence matrix"""
        n_pieces = fits[-1][0] + 1  # work out number of pieces from last of fits
        nrows = len(fits)
        ncols = len(self.triangles) + n_pieces
        m = np.zeros((nrows, ncols), dtype=bool)
        for i, fit in enumerate(fits):
            # here j refers to the piece
            j = len(self.triangles) + fit[0]
            m[i, j] = 1
            for j in fit[1]:
                # here j refers to the triangle
                m[i, j] = 1
        return m

    def covered(self, matrix):
        """List of fits which cover the individual triangles."""
        return [set(np.nonzero(matrix[:, i])[0]) for i, t in enumerate(self.triangles)]

    def calculate_equivalence(self):
        """matrix giving positions equivalent under point group"""
        equi = np.empty((48, 12), dtype=int)
        for i, t in enumerate(self.triangles):
            ts = self.point_group.apply_all(t)
            equi[i, :] = [self.trig_dict[c] for c in ts]
        self.equivalence = equi

    def equivalent_rolls(self, roll):
        """Return a set of dice rolls that are equivalent under point group"""
        idxs = np.array(roll) - 1
        x = self.equivalence[idxs, :] + 1
        rolls = x.T.tolist()
        [r.sort() for r in rolls]
        return rolls

    def equivalent_roll(self, roll):
        """Return the first of the set of rolls equivalent under point group"""
        b = self.equivalent_rolls(roll)
        b.sort()
        return b[0]


class PointGroup:
    """The point group symmetry of the triangular grid.
    (Dihedral(6), symmetries of a hexagon)"""

    def __init__(self, shift=(0, 0, 0)):
        self.shift = shift

        # matrix representing rotation by 60 degrees
        r = np.array(
            [[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 1], [0, 0, 0, 1]], dtype=int
        )
        # matrix representing reflection in `a` direction
        m = np.array(
            [[-1, 0, 0, 1], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=int
        )
        self.generators = [r, m]
        rotations = [np.linalg.matrix_power(r, n) for n in range(6)]
        reflections = [m @ x for x in rotations]
        self.elements = rotations + reflections

    def apply(self, t, idx):
        """Apply the symmetry element idx to triangle t"""
        s = self.shift
        v = np.array([t[0] - s[0], t[1] - s[1], t[2] - s[2], 1], dtype=int)
        x = np.dot(self.elements[idx], v)
        return (x[0] + s[0], x[1] + s[1], x[2] + s[2])

    def apply_all(self, t):
        """Apply all the group symmetry elements to triangle t"""
        return [self.apply(t, idx) for idx in range(12)]


class Dice:
    """The seven genius star dice"""

    def __init__(self):
        self.dice_numbers = [
            [1, 5, 15, 34, 44, 48],
            [2, 4, 7, 8, 9, 11, 16, 17],
            [10, 27, 31],
            [12, 13, 23, 24, 32, 33, 41, 42],
            [18, 22, 39],
            [19, 20, 21, 28, 29, 30],
            [25, 26, 36, 37, 38, 40, 45, 47],
        ]

    def roll(self, sort=True):
        """A random roll of the dice"""
        r = [random.sample(n, 1)[0] for n in self.dice_numbers]
        if sort:
            r.sort()
        return r

    def all_rolls(self):
        """Iterable giving all possible rolls"""
        return product(*self.dice_numbers)

    def rollable(self, roll):
        """Check if roll is possible with dice"""
        rollable = True
        current_roll = list(roll)
        for d in self.dice_numbers:
            i = set(current_roll).intersection(set(d))
            if len(i) != 1:
                # must have one number on each dice
                rollable = False
                break
            current_roll.remove(i.pop())
        return rollable


point_group = PointGroup()


class Piece:
    """Pieces described by a collection of triangles and a colour"""

    def __init__(self, triangles, col="black"):
        self.col = col

        # Look at all rotations / reflections of piece
        all_triangles = [
            [point_group.apply(t, i) for t in triangles] for i in range(12)
        ]

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
        self.dice = Dice()
        self.board = Board()
        self.original_pieces = Pieces(star=False)
        self.star_pieces = Pieces(star=True)

        self.original_fits = self.board.fits(self.original_pieces)
        self.star_fits = self.board.fits(self.star_pieces)

        self.original_matrix = self.board.incidence_matrix(self.original_fits)
        self.star_matrix = self.board.incidence_matrix(self.star_fits)

        self.original_covered = self.board.covered(self.original_matrix)
        self.star_covered = self.board.covered(self.star_matrix)

        if roll:
            self.new_roll(roll, star=star)

    def new_roll(self, roll, star=True):
        """Given a roll of the dice, block the appropriate triangles"""
        self.roll = roll
        self.star = star
        if star:
            self.pieces = self.star_pieces
            self.fits = self.star_fits
            self.matrix = self.star_matrix
            self.covered = self.star_covered
        else:
            self.pieces = self.original_pieces
            self.fits = self.original_fits
            self.matrix = self.original_matrix
            self.covered = self.original_covered

    def masks(self):
        """Boolean arrays showing which rows and columns of the board
        incidence matrix need to be removed given the blocker positions"""
        # columns to block are just correspond to the roll
        col_mask = np.zeros(self.matrix.shape[1], dtype=bool)
        col_mask[np.array(self.roll) - 1] = True

        # rows to block are those whose fits cover the blocked triangles
        blocked_set = set.union(*[self.covered[i - 1] for i in self.roll])
        row_mask = np.zeros(self.matrix.shape[0], dtype=bool)
        row_mask[list(blocked_set)] = True

        return ~row_mask, ~col_mask

    def incidence_matrix(self):
        """The incidence matrix corresponding to this game"""
        row_mask, col_mask = self.masks()
        ixgrid = np.ix_(row_mask, col_mask)
        return self.matrix[ixgrid]

    def plot_matrix(self):
        """Show the incidence matrix"""
        m = self.incidence_matrix()
        plt.figure(figsize=(5, 8))
        plt.spy(m, aspect="auto")

    def n_solutions(self):
        """Number of possible solutions"""
        m = self.incidence_matrix()
        return sum(1 for _ in covers_bool(m))

    def solutions(self):
        """Generator that yields all solutions"""
        m = self.incidence_matrix()
        row_mask, _ = self.masks()
        sub = np.nonzero(row_mask)[0]
        for cover in covers_bool(m):
            yield Solution([sub[i] for i in cover], self)

    def solve(self):
        """Solve game using an exact cover problem solver"""
        m = self.incidence_matrix()
        row_mask, _ = self.masks()
        sub = np.nonzero(row_mask)[0]
        solutions = self.solutions()

        try:
            solution = next(solutions)
        except StopIteration:
            if self.star:
                # no star solution, try solving without star
                self.new_roll(self.roll, star=False)
                return self.solve()
            else:
                # no solution at all
                raise NoSolution
        return solution

    def random(self, star=True, rollable=False):
        """Generate a random solvable puzzle"""
        not_solved = True
        while not_solved:
            if rollable is True:
                roll = self.dice.roll()
            else:
                roll = random.sample(range(1, 49), k=7)
                roll.sort()
            self.new_roll(roll)
            try:
                self.solve()
            except NoSolution:
                continue
            if star and not self.star:
                continue
            if self.star and not star:
                continue
            not_solved = False
        return self.roll

    def plot(self, show=True, numbers=True):
        self.board.plot(numbers=numbers)

        blocked_triangles = [self.board.triangles[i - 1] for i in self.roll]
        for t in blocked_triangles:
            plot_block([t], "white", symbol=True)

        if show:
            plt.show()


class NoSolution(Exception):
    pass


class Solution:
    """A solved game"""

    def __init__(self, solution, game):
        self.solution = solution
        self.game = game

    def plot(self, show=True):
        self.game.plot(numbers=False, show=False)

        for s in self.solution:
            f = self.game.fits[s]
            piece_idx, trig_idx = f[0], f[1]
            trigs = [self.game.board.triangles[i] for i in trig_idx]
            col = self.game.pieces[piece_idx].col
            plot_block(trigs, col)

        if show:
            plt.show()

    def solved(self):
        return len(self.solution) > 0


class Roll:
    """Simple class for storing the current dice roll"""

    def __init__(self, initial_roll):
        self.assign(initial_roll)

    def assign(self, roll):
        for i, r in enumerate(roll):
            setattr(self, "dice" + str(i), r)
        self.n_dice = len(roll)

    def as_list(self):
        return [getattr(self, "dice" + str(i)) for i in range(self.n_dice)]


def gui(run=True):
    """Very simple web-based GUI for the Genius Star Solver"""
    from nicegui import ui

    @ui.page("/", title="The Genius Star Solver")
    def page():
        ui.add_head_html(
            """
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <meta name="mobile-web-app-capable" content="yes">
            <meta name="description" content="Solver for The Genius Star Puzzle made by the Happy Puzzle Company" />
            <meta name="author" content="John F. Rudge" />
            """
        )

        with ui.column().classes("items-center"):
            dice = Dice()
            game = Game()
            roll = Roll(dice.roll(sort=False))

            fig = ui.pyplot(figsize=(3.0, 3.0), close=False)
            fig.classes("max-w-xs")

            def update():
                with fig:
                    game.new_roll(roll.as_list())
                    solution = game.solve()
                    plt.gca().clear()
                    solution.plot(show=False)

            update()

            for j, dn in enumerate(dice.dice_numbers):
                d = {x: str(x) for x in dn}
                tog = ui.toggle(d, on_change=update)
                tog.classes("max-w-xs")
                tog.bind_value(roll, "dice" + str(j))
                tog.props("padding=x xs")

            ui.link(
                "The Genius Star Solver on GitHub",
                "https://github.com/johnrudge/genius_star",
            )

    if run:
        ui.run()
