import random
import math
import colorama

X_POS = (1, 0, 0)
X_NEG = (-1, 0, 0)
Y_POS = (0, 1, 0)
Y_NEG = (0, -1, 0)
Z_POS = (0, 0, 1)
Z_NEG = (0, 0, -1)
DIRS = [X_POS, X_NEG, Y_POS, Y_NEG, Z_POS, Z_NEG]


class CompatibilityOracle(object):

    """The CompatibilityOracle class is responsible for telling us
    which combinations of tiles and directions are compatible. It's
    so simple that it perhaps doesn't need to be a class, but I think
    it helps keep things clear.
    """

    def __init__(self, data):
        self.data = data

    def check(self, tile1, tile2, direction):
        return (tile1, tile2, direction) in self.data


class Wavefunction(object):
    """The Wavefunction class is responsible for storing which tiles
    are permitted and forbidden in each location of an output image.
    """

    @staticmethod
    def mk(size, weights):
        """Initialize a new Wavefunction for a grid of `size`,
        where the different tiles have overall weights `weights`.

        Arguments:
        size -- a 3-tuple of (width, height)
        weights -- a dict of tile -> weight of tile
        """
        coefficients = Wavefunction.init_coefficients(size, weights.keys())
        return Wavefunction(coefficients, weights)

    @staticmethod
    def init_coefficients(size, tiles):
        """Initializes a 3-D wavefunction matrix of coefficients.
        The matrix has size `size`, and each element of the matrix
        starts with all tiles as possible. No tile is forbidden yet.

        NOTE: coefficients is a slight misnomer, since they are a
        set of possible tiles instead of a tile -> number/bool dict. This
        makes the code a little simpler. We keep the name `coefficients`
        for consistency with other descriptions of Wavefunction Collapse.

        Arguments:
        size -- a 2-tuple of (width, height, depth)
        tiles -- a set of all the possible tiles

        Returns:
        A 3-D matrix in which each element is a set
        """
        coefficients = []

        for x in range(size[0]):
            level = []
            for y in range(size[1]):
                row = []
                for z in range(size[2]):
                    row.append(set(tiles))
                level.append(row)
            coefficients.append(level)
        return coefficients

    def __init__(self, coefficients, weights):
        self.coefficients = coefficients
        self.weights = weights

    def get(self, co_ords):
        """Returns the set of possible tiles at `co_ords`"""
        x, y, z = co_ords
        return self.coefficients[x][y][z]

    def get_collapsed(self, co_ords):
        """Returns the only remaining possible tile at `co_ords`.
        If there is not exactly 1 remaining possible tile then
        this method raises an exception.
        """
        opts = self.get(co_ords)
        assert(len(opts) == 1)
        return next(iter(opts))

    def get_all_collapsed(self):
        """Returns a 3-D matrix of the only remaining possible
        tiles at each location in the wavefunction. If any location
        does not have exactly 1 remaining possible tile then
        this method raises an exception.
        """
        width = len(self.coefficients)
        height = len(self.coefficients[0])
        depth = len(self.coefficients[0][0])

        collapsed = []
        for x in range(width):
            layer = []
            for y in range(height):
                row = []
                for z in range(depth):
                    row.append(self.get_collapsed((x, y, z)))
                layer.append(row)
            collapsed.append(layer)
        return collapsed

    def shannon_entropy(self, co_ords):
        """Calculates the Shannon Entropy of the wavefunction at
        `co_ords`.
        """
        x, y, z = co_ords

        sum_of_weights = 0
        sum_of_weight_log_weights = 0
        for opt in self.coefficients[x][y][z]:
            weight = self.weights[opt]
            sum_of_weights += weight
            sum_of_weight_log_weights += weight * math.log(weight)

        return math.log(sum_of_weights) - (sum_of_weight_log_weights / sum_of_weights)


    def is_fully_collapsed(self):
        """Returns true if every element in Wavefunction is fully
        collapsed, and false otherwise.
        """
        for x, layer in enumerate(self.coefficients):
            for y, row in enumerate(layer):
                for z, cell in enumerate(row):
                    if len(cell) > 1:
                        return False
        return True

    def collapse(self, co_ords):
        """Collapses the wavefunction at `co_ords` to a single, definite
        tile. The tile is chosen randomly from the remaining possible tiles
        at `co_ords`, weighted according to the Wavefunction's global
        `weights`.

        This method mutates the Wavefunction, and does not return anything.
        """
        x, y, z = co_ords
        opts = self.coefficients[x][y][z]
        valid_weights = {tile: weight for tile, weight in self.weights.items() if tile in opts}

        total_weights = sum(valid_weights.values())
        rnd = random.random() * total_weights

        chosen = None
        for tile, weight in valid_weights.items():
            rnd -= weight
            if rnd < 0:
                chosen = tile
                break

        self.coefficients[x][y][z] = set([chosen])

    def constrain(self, co_ords, forbidden_tile):
        """Removes `forbidden_tile` from the list of possible tiles
        at `co_ords`.

        This method mutates the Wavefunction, and does not return anything.
        """
        x, y, z = co_ords
        self.coefficients[x][y][z].remove(forbidden_tile)



class Model(object):

    """The Model class is responsible for orchestrating the
    Wavefunction Collapse algorithm.
    """

    def __init__(self, output_size, weights, compatibility_oracle):
        self.output_size = output_size
        self.compatibility_oracle = compatibility_oracle

        self.wavefunction = Wavefunction.mk(output_size, weights)

    def run(self):
        """Collapses the Wavefunction until it is fully collapsed,
        then returns a 3-D matrix of the final, collapsed state.
        """
        while not self.wavefunction.is_fully_collapsed():
            self.iterate()

        return self.wavefunction.get_all_collapsed()

    def iterate(self):
        """Performs a single iteration of the Wavefunction Collapse
        Algorithm.
        """
        # 1. Find the co-ordinates of minimum entropy
        co_ords = self.min_entropy_co_ords()
        # 2. Collapse the wavefunction at these co-ordinates
        self.wavefunction.collapse(co_ords)
        # 3. Propagate the consequences of this collapse
        self.propagate(co_ords)

    def propagate(self, co_ords):
        """Propagates the consequences of the wavefunction at `co_ords`
        collapsing. If the wavefunction at (x,y) collapses to a fixed tile,
        then some tiles may not longer be theoretically possible at
        surrounding locations.

        This method keeps propagating the consequences of the consequences,
        and so on until no consequences remain.
        """
        stack = [co_ords]

        while len(stack) > 0:
            cur_coords = stack.pop()
            # Get the set of all possible tiles at the current location
            cur_possible_tiles = self.wavefunction.get(cur_coords)

            # Iterate through each location immediately adjacent to the
            # current location.
            for d in valid_dirs(cur_coords, self.output_size):
                other_coords = (cur_coords[0] + d[0], cur_coords[1] + d[1], cur_coords[2] + d[2])

                # Iterate through each possible tile in the adjacent location's
                # wavefunction.
                for other_tile in set(self.wavefunction.get(other_coords)):
                    # Check whether the tile is compatible with any tile in
                    # the current location's wavefunction.
                    other_tile_is_possible = any([
                        self.compatibility_oracle.check(cur_tile, other_tile, d) for cur_tile in cur_possible_tiles
                    ])
                    # If the tile is not compatible with any of the tiles in
                    # the current location's wavefunction then it is impossible
                    # for it to ever get chosen. We therefore remove it from
                    # the other location's wavefunction.
                    if not other_tile_is_possible:
                        self.wavefunction.constrain(other_coords, other_tile)
                        stack.append(other_coords)

    def min_entropy_co_ords(self):
        """Returns the co-ords of the location whose wavefunction has
        the lowest entropy.
        """
        min_entropy = None
        min_entropy_coords = None
        width, height, depth = self.output_size

        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if len(self.wavefunction.get((x,y, z))) == 1:
                        continue

                    entropy = self.wavefunction.shannon_entropy((x, y, z))
                    # Add some noise to mix things up a little
                    entropy_plus_noise = entropy - (random.random() / 1000)
                    if min_entropy is None or entropy_plus_noise < min_entropy:
                        min_entropy = entropy_plus_noise
                        min_entropy_coords = (x, y, z)

        return min_entropy_coords


def render_colors(matrix, colors):
    """Render the fully collapsed `matrix` using the given `colors.

    Arguments:
    matrix -- 2-D matrix of tiles
    colors -- dict of tile -> `colorama` color
    """
    for layer in matrix:
        for row in layer:
            output_row = []
            for val in row:
                color = colors[val]
                output_row.append(color + val + colorama.Style.RESET_ALL)

            print("".join(output_row))
        print("")


def valid_dirs(cur_co_ord, matrix_size):
    """Returns the valid directions from `cur_co_ord` in a matrix
    of `matrix_size`. Ensures that we don't try to take step to the
    left when we are already on the left edge of the matrix.
    """
    x, y, z = cur_co_ord
    width, height, depth = matrix_size
    dirs = []

    if x > 0: dirs.append(X_NEG)
    if x < width - 1: dirs.append(X_POS)
    if y > 0: dirs.append(Y_NEG)
    if y < height - 1: dirs.append(Y_POS)
    if z > 0: dirs.append(Z_NEG)
    if z < depth - 1: dirs.append(Z_POS)

    return dirs


def parse_example_matrix(matrix):
    """Parses an example `matrix`. Extracts:

    1. Tile compatibilities - which pairs of tiles can be placed next
        to each other and in which directions
    2. Tile weights - how common different tiles are

    Arguments:
    matrix -- a 3-D matrix of tiles

    Returns:
    A tuple of:
    * A set of compatibile tile combinations, where each combination is of
        the form (tile1, tile2, direction)
    * A dict of weights of the form tile -> weight
    """
    compatibilities = set()
    matrix_width = len(matrix)
    matrix_height = len(matrix[0])
    matrix_depth = len(matrix[0][0])

    weights = {}

    for x, layer in enumerate(matrix):
        for y, row in enumerate(layer):
            for z, cur_tile in enumerate(row):
                if cur_tile not in weights:
                    weights[cur_tile] = 0
                weights[cur_tile] += 1

                for d in valid_dirs((x, y, z), (matrix_width, matrix_height, matrix_depth)):
                    other_tile = matrix[x + d[0]][y + d[1]][z + d[2]]
                    compatibilities.add((cur_tile, other_tile, d))

    return compatibilities, weights


input_matrix = [
    [
        ['L','L','L','L','L','L'],
        ['L','L','L','L','L','L'],
        ['L','L','L','L','L','L'],
        ['L','C','C','L','L','L'],
        ['C','S','S','C','C','C'],
        ['S','S','S','S','S','S'],
        ['S','S','S','S','S','S'],
    ],
    [
        ['A','T','T','A','A','A'],
        ['T','A','T','A','A','A'],
        ['A','T','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
    ],
    [
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
        ['A','A','A','A','A','A'],
    ]
]

compatibilities, weights = parse_example_matrix(input_matrix)
compatibility_oracle = CompatibilityOracle(compatibilities)

colors = {
    'L': colorama.Fore.GREEN,
    'S': colorama.Fore.BLUE,
    'C': colorama.Fore.YELLOW,
    'A': colorama.Fore.CYAN,
    'B': colorama.Fore.MAGENTA,
    'T': colorama.Fore.RED,
}

for i in range(100):
    try:
        model = Model((3, 10, 50), weights, compatibility_oracle)
        output = model.run()
        render_colors(output, colors)
        break
    except:
        pass
