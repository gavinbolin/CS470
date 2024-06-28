from __future__ import print_function
import sys
import time
import collections
from operator import itemgetter

# Code for AI Class Sudoku Programming Assignment
# Written by Chris Archibald
# archibald@cs.byu.edu
# Last Updated February 5, 2020

def student_name():
    """
    This function returns your name.  This will be used for automated grading
    """

    #Task 1 CODE HERE

    #Change to be your name
    return 'Gavin Bolin'

class Cell():
    """
    This class represents a variable corresponding to an 
    individual cell in the Sudoku puzzle
    It's domain is the set of digits it could be (1-9)

    """

    def __init__(self):

        # The domain contains a list of the values that this variable could take on
        self.domain = list(range(1,10))     
        
        # The value assigned to this variable if it has been assigned
        # None means that it hasn't been assigned
        self.value = None


    def assign_value(self, value):
        """
        Assign [value] to be this variable's value
        This function sets the domain to be only this value, and sets the value to be the value
        """
        self.domain = [value]
        self.value = value

    def set_domain(self, domain):
        """
        Set the domain of this variable to be [domain]
        Set the value to unassigned (= None)
        """
        self.domain = domain[:]
        self.value = None

    def remove_value(self, value):
        """
        Remove [value] from this variable's domain
        Return False if the domain for this variable is now empty
        Return True if there is still at least one variable in the domain
        """
        if value in self.domain:
            self.domain.remove(value)
        if len(self.domain) == 0:
            return False
        return True

    def copy_cell(self, other):
        """
        Set the domain and value of this cell to that of the input other cell
        """
        self.domain = other.domain[:]
        self.value = other.value

class Sudoku():
    """
    This class contains all of the cells for an entire Sudoku puzzle
    """

    def __init__(self):
        """
        Initialize this puzzle by creating all of the cells (initially empty)
        """
        self.cells = [[ Cell() for j in range(9)] for i in range(9) ]

    def copy_puzzle(self, other):
        """
        Set all of this puzzle's cells to be the same as the other puzzle (input to function)
        """
        [[ self.cells[j][i].copy_cell(other.cells[j][i]) for j in range(9)] for i in range(9) ]

    def to_string(self):
        """
        Return a single string representation of the puzzle. For storage purposes
        """
        output_string = ''
        for r in range(9):
            for c in range(9):
                if self.cells[r][c].value:
                    output_string += str(self.cells[r][c].value)
                else:
                    output_string += '.'
        return output_string

    def print_puzzle(self):
        """
        Display the puzzle to the output, in human readable form
        """
        for r in range(9):
            if (r % 3) == 0 and r > 0:
                print('')
            for c in range(9):
                if (c % 3) == 0 and c > 0:
                    print(' ', end="")
                if self.cells[r][c].value is None:
                    print('.', end="")
                else:
                    print(self.cells[r][c].value, end="")
            print('')

    def is_solved(self):
        """
        Return true if every cell has been assigned a value 
        This assumes that only valid assignments have been made
        """
        for r in range(9):
            for c in range(9):
                if self.cells[r][c].value is None:
                    return False
        return True

    def input_puzzle(self, puzzle_string):
        """
        Initialize this puzzle to match the starting state specified by [puzzle string]
        This is done by assigning all of the cells to be their given value (if specified)
        Then forward-checking is performed for every variable assingment that has been made
        """

        # Cycle through the puzzle and assign all the cells that are given in the input string
        r = 0
        c = 0
        for v in puzzle_string:
            if v != '.':
                self.cells[r][c].assign_value(int(v))
            c += 1
            if c > 8:
                c = 0
                r += 1

        # Cycle through the puzzle and perform forward checking for every variable that 
        # was assigned a value
        for r in range(9):
            for c in range(9):
                if self.cells[r][c].value is not None:
                    success = self.forward_check(r, c, self.cells[r][c].value)
                    if not success:
                        ### This puzzle has some serious problem.  Exit now. 
                        print('Contradiction in Puzzle! Invalid input. Exiting')
                        print(' Puzzle string: ', puzzle_string)
                        sys.exit()

    def forward_check(self, r, c, value, mode='remove'):
        '''
        Perform forward checking for the assignment of [value] to the cell at [row], [column]
        If [mode] == remove, then values should actually be removed other cell domains. 
        Use the remove_value() function from the Cell class.
        Return False if any domain is empty as a result of forward checking, True otherwise.
        If [mode] == count, then values shouldn't be removed, but merely a count returned
        of how many values forward checking would remove
        '''

        # TASK 2 Code here

        if mode == 'remove':
            row = True
            col = True
            box = True
            for i in range(9):
                # if len(self.cells[i][c].domain) == 1 and self.cells[i][c].value is None:
                #     self.cells[i][c].value = self.cells[i][c].domain[0]
                #     self.cells[i][c].domain = [value]
                if self.cells[i][c].value is None:
                    row = self.cells[i][c].remove_value(value)
                if not row:
                    # print('EXIT HERE::1', r, c)
                    return row

                # if len(self.cells[r][i].domain) == 1 and self.cells[r][i].value is None:
                #     self.cells[r][i].value = self.cells[r][i].domain[0]
                #     self.cells[r][i].domain = [value]
                if len(self.cells[r][i].domain) != 1 and self.cells[r][i].value is None:
                    col = self.cells[r][i].remove_value(value)
                if not col:
                    # print('EXIT HERE::2', r, c)
                    return col

                grid, cell = self.get_grid_cell(r, c)
                i_r, i_c = self.get_row_column(grid, i)
                # if len(self.cells[i_r][i_c].domain) == 1 and self.cells[i_r][i_c].value is None:
                #     self.cells[i_r][i_c].value = self.cells[i][c].domain[0]
                #     self.cells[i_r][i_c].domain = [value]
                if self.cells[i_r][i_c].value is None:
                    box = self.cells[i_r][i_c].remove_value(value)
                if not box:
                    # print('EXIT HERE::3',r,c)
                    return box
            return True
        elif mode == 'count':
            count = 0
            for i in range(9):
                if self.cells[i][c].domain.count(value) > 0:
                    count += 1
                if self.cells[r][i].domain.count(value) > 0:
                    count += 1
                grid, cell = self.get_grid_cell(r, c)
                i_r, i_c = self.get_row_column(grid, i)
                if self.cells[i_r][i_c].domain.count(value) > 0:
                    count += 1
            return count

    def get_row_column(self, grid, cell):
        '''
        Return the row and column indices for a given [grid] and [cell] location
        '''
        base_r = 3*int(grid / 3)
        base_c = 3*(grid % 3)
        off_r = int(cell / 3)
        off_c = cell % 3
        return base_r + off_r, base_c + off_c

    def get_grid_cell(self, row, column):
        '''
        Return the grid and cell indices for a given [row] and [column] location
        '''
        grid = 3*int(row / 3) + int(column / 3)
        base_r = 3*int(grid / 3)
        base_c = 3*(grid % 3)
        off_r = row - base_r
        off_c = column - base_c
        cell = 3*int(off_r) + int(off_c)
        return grid, cell

# Other functions (not in Sudoku class)
# When [puzzle] is an input 
# it is an object of the Sudoku puzzle class [Sudoku]

def mrv(puzzle, unassigned):
    '''
    Return a list of the (row, column) tuples, out of the list of [unassigned] cells given
    that have the minimum remaining values 
    [unassigned] is a list of (row, column) tuples corresponding to cell locations
    '''
    if len(unassigned) == 1 or len(unassigned) == 2:
        return unassigned
    least = 9
    temp = []
    # for i in unassigned:
    #     for num in range(1,10):
    #         puzzle.forward_check(i[0], i[1], num, 'remove')
    for i in unassigned:
        for num in range(1, 10):
            remainder = puzzle.forward_check(i[0], i[1], num, 'count')
            if (remainder < least):
                least = remainder
                temp.clear()
                temp.append(i)
            elif (remainder == least):
                temp.append(i)
    return temp

def max_degree(puzzle, tied):
    '''
    Return a list of the (row, column) tuples, out of the list of [tied] cells (according to mrv) given
    that have the maximum degree, i.e. are connected to the most unassigned other variables
    This uses the count_constraints function
    '''
    max_c = 0
    mc_variables = []
    for t in tied:
        cur_c = count_constraints(puzzle, t[0], t[1])
        if cur_c == max_c:
            mc_variables.append(t)
        if cur_c > max_c:
            max_c = cur_c
            mc_variables = [t]

    return mc_variables


def count_constraints(puzzle, r, c):
    '''
    Count the number of unassigned variables that are in the same row, column and grid as 
    the input location [row], [column]
    return this number
    This is called by the max_degree function
    '''
    # TASK 3 CODE HERE
    count = 0
    for i in range(9):
        if puzzle.cells[i][c] == None:
            count += 1
        if puzzle.cells[r][i] == None:
            count += 1
        grid, cell = puzzle.get_grid_cell(r, c)
        if puzzle.get_row_column(grid, cell) == None:
            count += 1
    return count

def get_unassigned_variables(puzzle):
    '''
    Return a list of the variables in the puzzle that haven't been assigned a value
    yet. These will be (row, column) tuples
    '''
    unassigned = []
    for r in range(9):
        for c in range(9):
            #Unassigned cells have a value of None
            if puzzle.cells[r][c].value is None:
                unassigned.append((r,c))
    return unassigned

def select_variable(puzzle):
    '''
    This function will return the row and column location of the variable that should be
    selected next from [puzzle]. 
    It will do this using the minimum-remaining-values heuristic, with ties broken by the
    degree heuristic, and any remaining ties broken arbitrarily
    '''

    # 0. Get the initial list of all unassigned variables
    unassigned = get_unassigned_variables(puzzle)
    return unassigned[0][0], unassigned[0][1]
    
    # 1. Use MRV heuristic to get list of variables with the min remaining values
    minimum_remaining_values = mrv(puzzle, unassigned)

    r = minimum_remaining_values[0][0]
    c = minimum_remaining_values[0][1]
    mrv0 = puzzle.cells[r][c]
    if len(mrv0.domain) == 1:
        puzzle.cells[minimum_remaining_values[0][0]][minimum_remaining_values[0][1]].assign_value(mrv0.domain[0])
        puzzle.forward_check(r, c, mrv0.value, 'remove')

    # If MRV identifies a unique variable, then return it
    if len(minimum_remaining_values) == 1:
        return minimum_remaining_values[0][0], minimum_remaining_values[0][1]

    # 2. Refine list to those with maximum degree
    most_constaining_variables = max_degree(puzzle, minimum_remaining_values)

    # 3. Return first variable in the list.  This will be the only one if there was a
    #    unique most constraining variable, or the "alphabetically first" or "arbitrary" one of
    #    them if there were ties
    return most_constaining_variables[0][0], most_constaining_variables[0][1]

def order_values(puzzle, r, c):
    '''
    For the variable at the given row and column, return the values of this variables domain, 
    ordered by the least-constraining value heuristic.  You can use the forward_check() function 
    in the 'count' mode to count the number of values that would be removed from other variables'
    domains by a particular variable=value assignment
    '''

    #Get the current domain for this variable
    domain = puzzle.cells[r][c].domain[:]
    temp = {}
    for i in domain:
        temp[i] = puzzle.forward_check(r, c, i, 'count')
    temp = collections.OrderedDict(sorted(temp.items(), key=itemgetter(1)))
    return temp

def backtracking_search(puzzle):
    '''
    Perform a recursive backtracking search on [puzzle], 
    ensuring that a copy is made of the puzzle 
    before modifying it and recursing. 

    If unsuccessful, a None value is returned
    If successful, this function returns the solved puzzle object
    '''

    # PSEUDO-CODE to help with structuring this function

    # TASK 6 CODE HERE 

    # 1. Base case, is input [puzzle] solved? If so, return the puzzle. Use is_solved() function
    #    to see if the puzzle is solved.

    # 2. Select a variable to assign next ( use select_variable() function, which returns 
    #    row and column of the variable 

    # 3. Select an ordering over the values (use order_values(r,c) where r, c are the row
    #    and column of the selected variable.  It returns a list of values

    # 4. For each value in the ordered list:

        # 4.1 Get a copy of the puzzle to modify
        #     4.1.a Create new puzzle

        #     4.1.b Set it to be equal to the current puzzle (use copy_puzzle())

        # 4.2 Assign current value to selected variable (use assign_value())

        # 4.3 Forward check from this assignment (use forward_check(), in 'remove' mode)
        #     which will return False if this assignment is invalid (empty domain was found)
        #     or True if it is valid.

        # 4.4 If forward checking detects a problem, then continue to the next value

        # 4.5 If forward checking doesn't detect problem, then recurse on the
        #     modified puzzle (call backtracking_search())

        # 4.6 If the search succeeds (return value of backtracking is not None)
        #     return solved puzzle! (this is what backtracking_search should return)

        # 4.7 If search is a failure, continue with next value for this variable

        # 5. If all values for the chosen variable failed, return failure (None)
        # return None


    if puzzle.is_solved():
        print('CHECK VALID::', puzzle.cells[0][5].value)
        return puzzle

    r, c = select_variable(puzzle)
    ls = order_values(puzzle, r, c)
    for i in ls:
        new_p = Sudoku()
        new_p.copy_puzzle(puzzle)
        new_p.cells[r][c].assign_value(i)
        result = new_p.forward_check(r, c, i, 'remove')
        if result == False:
            continue
        else:
            bt_result = backtracking_search(new_p)
            if bt_result is not None:
                return bt_result
            else:
                continue
    return None

    
if __name__ == "__main__":

    # Check the input arguments (should just be the puzzle file)
    if len(sys.argv) != 2:
        print('USAGE: python pa2.py PUZZLE_FILE.txt')
    else:
        print('Searching for solutions to puzzles from file: ', sys.argv[1])
        
        # Open puzzle file
        pf = open(sys.argv[1],'r')
        
        # How many puzzles do we want to solve
        max_to_solve = 9
        
        # Variable to keep track of the number of puzzles solved
        num_solved = 0

        # Start time for solving            
        start_time = time.perf_counter()

        # Each puzzle takes up a single line in the file
        for ps in pf.readlines():
            print('\nSearching to find solution to following puzzle:', ps)

            # Create a new puzzle to solve
            p = Sudoku()
            
            # Initialize the puzzle to match the file
            p.input_puzzle( ps.rstrip() )
            
            # Display the puzzle (before it has been solved)
            p.print_puzzle()

            # Solve the puzzle 
            success_p = backtracking_search(p)

            # If the puzzle was solved (success_p is the puzzle, None if no solution)
            if success_p: 
                print('\n Successfully solved puzzle!  Here is solution:\n')
                
                # Display the solution to the puzzle
                success_p.print_puzzle()
                num_solved += 1
                # Update the statistics

            else:
                print('\n!! Unable to solve puzzle !!\n')
                break

            # Stop when we solve enough
            if num_solved > max_to_solve:
                break

        # Record end time 
        end_time = time.perf_counter()

        # Close the puzzle file
        pf.close()

        # Print Statistics
        print('\n**************\nSolved', num_solved, 'puzzles in', end_time - start_time, ' seconds. ')

        # Compute and display average solve time
        average_time = 0
        if num_solved > 0:
            average_time =  (end_time - start_time) / float(num_solved)
        print(' Average Solve Time = ', average_time, ' seconds')

