from __future__ import print_function
from queue import Queue
import sys
import math
import random
from pa1 import *

def breadth_first(depth,pq,visited,of,maxnum):
    num_saved = 0
    while not pq.empty():
        cur_node = pq.get()
        visited[cur_node.puzzle.id()] = True
        if cur_node.cost > depth:
            of.write(cur_node.puzzle.id() + '\n') 
            num_saved += 1
        else:
            moves = cur_node.puzzle.get_moves()
            random.shuffle(moves)
            for m in moves:
                np = Puzzle(cur_node.puzzle.state[:])
                np.do_move(m)
                if np.id ()not in visited:
                    new_node = SearchNode(cur_node.cost + 1, np, cur_node.path + m)
                    pq.put(new_node)
        if num_saved == maxnum:
            break
    print('Done creating puzzles.  Created ', num_saved)
        
if __name__ == '__main__':
    #Modify file name to create new file
    print('ARGS: ', sys.argv)
    if len(sys.argv) != 2:
        print("usage: python create_puzzle.py FILE_NAME_TO_SAVE")
        sys.exit()

    #Open up the file for saving
    of = open(sys.argv[1], 'w')

    #Parameters of generation
    max_number_per_gen = 500  #How many puzzles can each gen produce (max)? 
    min_solution_depth = 2    #What is the minimum solution depth we should consider
    max_solution_depth = 30   #What is the maximum solution depth we should consider
    num_gens = 2             #How many times should we generate puzzles ((each depth needs only be generated once)

    #Repeat the puzzle generating process, for different solution lengths
    for i in range(num_gens):
        #modify depth to create specific solution length puzzles
        depth = random.randint(min_solution_depth,max_solution_depth)
        print('Creating puzzles of length: ', depth)
        a = list(range(10))
        p = Puzzle(a)
        s = SearchNode(0,p,'')
        visited = dict()
        pq = Queue()
        pq.put(s)
        breadth_first(depth,pq,visited,of,max_number_per_gen)
    
    of.close()

    #Filter out duplicate puzzles that might have been created    
    pf = open(sys.argv[1], 'r')
    pez = []
    count = 0
    for ps in pf.readlines():
        count += 1
        if ps not in pez:
            pez.append(ps)

    print(count, 'puzzles originally created.')
    print('After duplicate removal,', len(pez), 'puzzles remain.')
    pf.close()

    of = open(sys.argv[1], 'w')
    for p in pez:
        of.write(p)
    of.close()

   
