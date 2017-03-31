# Back tracking SUDOKU

# List of blocks in the same row or same column or the 3 *3 block
adjacent = {}

for i in range(8):
  for j in range(8):
    row = set([(i, k) for k in range(8) if k != j])
    column = set([(k, j) for k in range(8)  if k != i])
    diagonal = set([])
    n = [(-1,1), (1,1), (-1,-1), (1, -1)]
    for k in range(1,9):
      for l in n:
        x, y = i + l[0] * k  , j + l[1] * k
        if x < 8 and x >=0 and y < 8 and y >=0:
          diagonal.add((x, y))
    adjacent[(i,j)] = (diagonal | row | column) - set([(i,j)])


def is_solution(the_input):
  return  sum([j for i in the_input for j in i if j == 1]) == 8

def process_solution(the_input):
  print the_input

def construct_candidates(the_input):
  for row, col in find_first_missing(the_input):
    if will_fit(the_input, row, col):
      yield (row, col)



def find_first_missing(the_input):
  for i in range(8):
    for j in range(8):
      if the_input[i][j] == 0:
        yield i,j


def will_fit(the_input, row, col):
  for i, j in adjacent[(row, col)]:
    if the_input[i][j] == 1:
      return False
  return True

def make_move(the_input, pos):
  i, j = pos
  the_input[i][j]  = 1

def unmake_move(the_input, pos):
  i, j = pos
  the_input[i][j]  = 0


def backtrack(the_input):
  global finished
  if is_solution(the_input):
    process_solution(the_input)
    print  'Setting finished %s' % the_input
    finished = True
  else:
    for i in construct_candidates(the_input):
      print 'Trying to fit  queen into (%s, %s) with current board %s' % (i[0], i[1], the_input)
      make_move(the_input, i)
      print the_input
      backtrack(the_input)
      if finished:
        return
      unmake_move(the_input, i)


def sudoku(board):
"""
pass the grid as a new line delimited string like 
sudoku('''00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000
''')
"""
board = """00000000
00000000
00000000
00000000
00000000
00000000
00000000
00000000"""
rows = board.split('\n')
grid = []
for i_index, i in enumerate(rows):
  grid.append([])
  for j_index, j in enumerate(list(i.strip())):
    grid[i_index].append(int(j))
print backtrack(grid)