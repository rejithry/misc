
from copy import deepcopy

def is_substring(matrix, string):
  for i in range(len(matrix)):
    for j in range(len(matrix[i])):
      if _is_substring(matrix, string, (i, j)):
        print 'result', matrix, (i, j)
        return True
  return False

def _is_substring(matrix, string, pos):
  print string, matrix, pos
  if len(string) < 1:
    return True
  if string[0] != matrix[pos[0]][pos[1]]:
    return False
  t = deepcopy(matrix)
  t[pos[0]][pos[1]] = -1
  for new_pos, char in next_pos(matrix, pos):
    if _is_substring(t, string[1:], new_pos):
      return True
  return False

def next_pos(matrix, pos):
  positions = [  (pos[0] - 1, pos[1]),  (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1) ]  
  for i in positions:
    if i[0] < 0 or i[0] > len(matrix) - 1 or i[1] < 0 or i[1] > len(matrix[i[0]]) - 1:
      continue
    if matrix[i[0]][i[1]] != -1:
      yield i, matrix[i[0]][i[1]]


matrix = [ ['a', 'b', 'c', 'e'], ['s', 'f', 'c', 's'], ['a', 'd', 'e', 'e'] ]
string = 'bcced'

is_substring(matrix, string)


## Back tracking

def goal_test(matrix, string):
  if len(string) == 0:
    return True
  else:
    return False

def next_candidates(matrix, pos, char):
  positions = [  (pos[0] - 1, pos[1]),  (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1) ]  
  for i in positions:
    if i[0] < 0 or i[0] > len(matrix) - 1 or i[1] < 0 or i[1] > len(matrix[i[0]]) - 1:
      continue
    if matrix[i[0]][i[1]] != -1 and matrix[i[0]][i[1]] == char:
      yield i[0], i[1]

def move(matrix, string, i , j):
  print 'Trying to fit char %s at (%s, %s) to first char of %s' % ( matrix[i][j], i, j, string)
  matrix[i][j] = -1
  string = string[1:]
  return string

def unmove(matrix, string, i , j, t, u):
  matrix[i][j] = t
  return  u + string

def is_substring(matrix, string, i, j):
  if goal_test(matrix, string):
    return True
  else:
    for i, j in next_candidates(matrix, (i, j), string[0]):
     t = matrix[i][j]
     u = string[0]
     string = move(matrix, string, i, j)
     if is_substring(matrix, string, i, j):
       return True
     string = unmove(matrix, string, i, j, t, u)


def is_substring_wrap(matrix, string):
  for i in range(len(matrix)):
    for j in range(len(matrix[i])):
      if is_substring(matrix, string, i, j):
        print 'result', matrix, (i, j)
        return True
  return False




## Robot

def next_candidates(x, y, m, n, k, visited):
  pos = (x,y)
  positions = [  (pos[0] - 1, pos[1]),  (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1) ]  
  for i in positions:
    if i[0] < 0 or i[0] > m - 1 or i[1] < 0 or i[1] > n- 1:
      continue
    if sum(map(int, list(str(i[0])))) + sum(map(int, list(str(i[1])))) <= k and not visited[i[0]][i[1]]:
      yield i[0], i[1]

s = 0

def explore(m, n, x, y, k, visited):
  for i, j in next_candidates(x, y, m, n, k, visited):
    if not visited[i][j]:
      print visited
      visited[i][j] = True
      print 'visiting %s, %s from %s, %s' % (i,j,x,y)
    explore(m, n, i, j, k, visited)
  
visited = []
for i in range(5):
  visited.append(bitarray(4))


explore(5, 4, 0, 0, 2, visited)




