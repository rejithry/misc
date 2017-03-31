def anagrams(l):
  d = {}
  for i in l:
    sorted_i = ''.join(sorted(i))
    if sorted_i in d:
      d[sorted_i].append(i)
    else:
      d[sorted_i] = [i]
  return d.values()

anagrams(['star', 'rats', 'ice', 'cie', 'arts'])

points([(1,2), (3,4), (10,10)], 2)


def partition(d, start, end):
  pivot = d[start][1]
  pivot_pos = start
  for index, element in enumerate(d[start + 1:end]):
    if element[1] < pivot:
      d[pivot_pos], d[start + index + 1] , d[pivot_pos + 1] = d[pivot_pos + 1], d[start + index + 1] , d[start + index + 1], 
  return pivot_pos


def points(points, k):
  l = len(points)
  #Find distances
  distances = []
  for i in points:
    distances.append((i, math.sqrt(i[0] * i[0] + i[1] * i[1] )))
  pos_k = 0
  i = 0
  j = l
  while True:
    
    x = partition(distances, i, j)
    if x == k - 1:
      pos_k = x
      break
    elif x < k -1:
      i = x + 1
    else:
      j = x
  distances[0], distances[pos_k] = distances[pos_k], distances[0]
  partition(distances, i, j)
  return distances[:k]


def substring(a, b):
  l = len(a)
  i = 0
  j = l
  if len(b) < l:
    return -1
  while True:
    print  b[i:j], a
    if b[i:j] == a:
      return True
    else :
      i += 1
      j += 1
    if j > len(b):
      return -1

def substring(a, b):
  l = len(a)
  i = 0
  if len(b) < l:
    return -1
  while True:
    if i > len(b):
      return -1
    if b[i] == a[0]:
      print b[i:i + l]
      if b[i:i + l] == a:
        return True
      else:
        i += 1
    else:
      i += 1

def sub_sequence(a):
  i = 0
  j = 1
  r = []
  a = sorted(a)
  while True:
    if  i > len(a) or j > len(a) or i:
      return r
    s = sum(a[i: j])
    if s == 0:
      r.append(a[i: j])
      i += 1
    elif s < 0:
      j += 1
    else:
      i += 1


bcd
a 25* 25
b a 25
b c [a  b d]

def lexic(s):
  pos = [ord(i) - 96 for i in list(s)]
  print pos
  t = 0
  for index, i in enumerate(pos):
    if index == 0:
      t += (i - 1) * pow(25, len(s) - index - 1)
    elif index == len(s) -1:
      if pos[index -1] == i:
        t += (i - 1) * pow(25, len(s) - index - 1)
      else:
        t += (i - 2) * pow(25, len(s) - index - 1)      
    else:
      if pos[index -1] == i:
        t += (i - 1) * pow(25, len(s) - index - 1)
      else:
        t += (i - 2) * pow(25, len(s) - index - 1)
    print t, i, index
  return t % 1009


import sys


def find_min(pal, s1, s2):
  if pal:
    if s1 < pal[0]:
      if s1 < s2:
        pal[0] = s1
      else:
        pal[0] = s2
    elif s2 < pal[0]:
      if s2 < s1:
        pal[0] = s2
      else:
        pal[0] = s1
  else:
    if s1 < s2:
      pal.append(s1)
    else:
      pal.append(s2)

def find_pal(a,b):
    pals = []
    
    i = 0
    j = len(a)
    k = len(a)
    
    
    if not set.intersection(set(list(a)), set(list(b))):
        return -1
    if j == k and j == 1:
        return [[j[0], j[0]]]
    pals = []
    max_len = 3
    while True:
        if i == j:
            break
        if pals and len(a[i:j]) < len(pals[0]):
          break
        
        if a[i:j] in b:
            s1 = a[i:j + 1] +  b[i:j]
            s2 = a[i:j] +  b[i-1:j]
            find_min(pals, s1, s2)
        else:
            if j + 1 > len(a):
                k = k -1
                i = 0
                j = k
            else:
                i = i + 1
                j = j + 1
    return pals
    
for i in  range(int(sys.stdin.readline().strip())):
    a = sys.stdin.readline().strip()
    b = sys.stdin.readline().strip()
    print find_pal(a,b)
    print '#########'



def find_longest_palyndrome(substring_pos, s, pos):
  if pos == 'after':
    for i in range()

1 2 199 200 300 150 200
1 2 100 150 200
1 2 3 4


def bin(l, a, start = None, end = None):
  if len(l) == 1:
    return abs(l[0] - a)
  if not start:
    start = 0
  if not end:
    end = len(l) -1
  if start == end:
    return abs(l[start] - a)
  if start + 1 == end:
    return min(abs(l[start] - a), abs(l[end] - a))    
  cur = (start - end) /2
  if a == l[cur]:
    return 0
  elif a < l[cur]:
    return bin(l, a, start = start, end = end - ((end - start) /2))
  elif a < l[cur]:
    return bin(l, a, start = start + ((end - start) /2), end = end )


alphabet1 = { i + 1 : chr(i + 97) for i in range(26) }
alphabet2 = { chr(i + 97) : i + 1 for i in range(26) }
def comb(s):
  if len(s) == 1:
    yield alphabet1[int(s[0])]
  elif len(s) == 2:
    if int(s) < 27:
      yield alphabet1[int(s[0])] + alphabet1[int(s[1])]
      yield alphabet1[int(s)]
    else:
      yield alphabet1[int(s[0])] + alphabet1[int(s[1])]
  else:
    if int(s[:2]) < 27:
      for i in 

      (s[1:]):
        yield alphabet1[int(s[0 ])] + i
      for i in comb(s[2:]):
        yield alphabet1[int(s[:2])] + i
    else:
      for i in comb(s[1:]):
        yield alphabet1[int(s[0 ])] + i


def sort(a):
  for index,i in enumerate(a):
    if i == 1 and index != 0:
      a[index], a[0], a[1]



def unique(a):
  s = set(a)
  for i in list(a):
    s2 = s - set([i])
    for j in list(s2):
      s3 = s2 - set([j])
      for k in s3:
        yield (i, j, k)
      s2.discard(j)
    s.discard(i)



class Tree:
  def __init__(self):
    self.root = None
  def add(self, e, root = None, distance = 0):
    root  = root or self.root
    if not self.root:
      self.root = Node(e)
      self.root.distance = distance
    elif e < root.a:
      if not root.left:
        root.add_left(e, distance -1)
      else:
        self.add(e, root.left, distance -1)
    else:
      if not root.right:
        root.add_right(e, distance + 1)
      else:
        self.add(e, root.right, distance + 1)
  def serialize(self, root = None,current = ''):
    root = root or self.root
    left_tree = self.serialize(root = root.left) if root.left else ''
    right_tree = self.serialize(root = root.right) if root.left else ''
    return str(root.a) + '->' + left_tree + '<-' + right_tree
  def traverse(self, root = None):
    root = root or self.root
    self.traverse(root.left) if root.left else None
    print str(root.a) + ':' + str(root.distance)
    self.traverse(root.right) if root.right else None


def add(g, i, j, b):
  e = b[i][j]
  if e not in g:
    g[e] = []
  indices = [(i + 1, j), (i + 1, j + 1), (i -1, j), (i - 1, j -1), (i, j + 1), (i, j -1 ), (i + 1, j -1), (i - 1, j + 1)]
  for k in indices:
    if k[0] >=0 and k[0] < 4 and k[1] >=0 and k[1] < 4:
      g[e].append(b[k[0]][k[1]])


def build(b):
  g = {}
  for i in range(4):
    for j in range(4):
      add(g, i, j, b)
  return g
b = [   list('SMEF'), list('RATD'), list('LONI'), list('KAFB')   ]
g = build(b)

def search(s):
  for index,i in enumerate(list(s)):
    if index == len(s) - 1:
      return True
    if i not in g or s[index + 1] not in g[i]:
      return False


def move(s):
  j = -1
  for index in range(len(s)):
    if j == -1 and s[index] == 0:
      j = index
    if s[index] > 0 and j < index and j != -1:
      s[index] , s[j] = s[j], s[index]
      j += 1
  return s


def reverse(s):
  return ' '.join(s.split(' ')[::-1])[::-1]

  for i in range(l/2):
    s[i], s[ l - i] = s[l - i], s[i]


class Median:
  def __init__(self):
    self.median = None
    self.


def one_edit_aprt(a, b):
  i = 0
  j = 0
  la = len(a)
  lb = len(b)
  edits = 0
  while True:
    if edits > 1:
      return False
    if i > la - 1:
      if edits == 0 and j == lb -1:
        return True
      elif edits == 1 and j > lb -1:
        return True
      else:
        return False
    if j > lb - 1:
      if edits == 0 and i == la -1:
        return True
      elif edits == 1 and i > la -1:
        return True
      else:
        return False
    if a[i] != b[j] and la > lb:
      edits += 1
      i += 1
    elif a[i] != b[j] and la < lb:
      edits += 1
      j += 1
    elif a[i] != b[j] and la ==lb:
      edits += 1
      j += 1
      i += 1
    else:
      j += 1
      i += 1

def goal_test(k, n):
  return k == n

def construct_candidates(cur, )

def backtrack(l):



a = { 1 : [2, 3], 2 : [1, 4], 3 : [1, 4], 4: [3,2, 5], 5: [4,6], 6 : [5, 7, 9], 7 : [6, 8], 8 : [7, 9], 9 : [8, 6] }
def bfs_shortest_distance(g, s ,e):
  st = []
  st.append((s, 0))
  visited = set([])
  while st:
    t = st.pop(0)
    if t[0] in visited:
      continue
    visited.add(t[0])
    if t[0] == e:
      return t[1]
    for i in g[t[0]]:
      if i not in visited:
        st.append((i, t[1] + 1))
  return -1

def bfs_shortest_path(g, s ,e):
  st = []
  st.append((s, [s]))
  visited = set([])
  while st:
    t = st.pop(0)
    if t[0] in visited:
      continue
    visited.add(t[0])
    if t[0] == e:
      return t[1]
    for i in g[t[0]]:
      if i not in visited:
        st.append((i, t[1] + [i]))
  return []


a = { 1 : [2, 3], 2 : [1, 4], 3 : [1, 4], 4: [3,2, 5], 5: [4,6], 6 : [5, 7, 9], 7 : [6, 8, 9], 8 : [7, 9], 9 : [8,7, 6] }
bfs_all_paths(a,1,9)
def bfs_all_paths(g, s ,e):
  st = []
  st.append((s, [s]))
  visited = 
  prev = -1
  while st:
    t = st.pop(0)
    if t[0] in visited:
      visited[t[0]].append(prev)
    else:
      visited[t[0]] = [prev]
    if t[0] == e:
      yield t[1]
    for i in g[t[0]]:
      if i not in visited or ( i in visited and t[0] not in visited[i]):
        if i not in t[1]:
          st.append((i, t[1] + [i]))
    prev = t[0]

def explore(g, current_path, vertext, end, start):
  if vertext == start and current_path:
    return
  if vertext == end:
    yield current_path + [vertext]
  elif vertext in g:
    for i in g[vertext]:
      for j in  explore(g, current_path + [vertext], i, end, start):
        yield j

def all_paths(g, s e):
  explore

a = { 1 : [2, 3, 4,7], 2 : [4], 7: [8, 10], 10: [4]}

class Node:
  def __init__(self, a = None):
    self.a = a
    self.term = True
    self.m = {}


class Trie:
  def __init__(self):
    self.root = Node()
  def add(self, a):
    cur = self.root
    for index,i in enumerate(a):
      if i in cur.m:
        cur = cur.m[i]
      else:
        n = Node()
        n.term = False
        cur.m[i] = n
        cur = cur.m[i]
    if len(cur.m) == 0:
      cur.term = True

00001011
00001000
    1011
1000
1001

N W S E N W

def is_cross(l):
  right = 0
  up = 0
  prev_right = 0
  prev_up = 0
  cross = False
  for index,i in enumerate(l):
    if index % 4 == 0:
      up += i
      if up > prev_up:
        cross = True
        break      
    elif index % 4 == 1:
      right += -1 * i
      if right < prev_right:
        cross = True
        break
    elif index % 4 == 2:
      up += -1 * i
      if up < prev_up:
        cross = True
        break  
    else:
      right += i
      if right > prev_right:
        cross = True
        break
  if right != prev_right:
    prev_right = right
  if up != prev_up:
    prev_up = up
  return cross


def add_array(l):
  nine_count = 0
  length = len(l)
  for i in reversed(range(len(l))):
    if l[i] == 9:
      nine_count += 1
    else:
      break
  if length == nine_count:
    return [1] + [0] * nine_count
  print nine_count
  if nine_count:
    return l[:length - nine_count -1] + [l[length - nine_count - 1 ] + 1]  + ([0] * nine_count)
  else:
    return l[:-1] + [l[-1] + 1]

[[1,0,4,....], [0,0,1.....],....]

def sudoku(a):


backtrack


1 2 3 4 5

123 234 345
12 23 34 45
1234 2345
12345
1 2 3 4 5

111
555


     1234


 1234
122334
123234
 1234



 12345
12233445
123234345
 12342345
  12345


# Back tracking for all subsets



def process_solution(a, k, the_input):
  print [ index + 1  for index, i  in enumerate(a) if i]

def is_solution(a, k, the_input):
  return len(a) == k


def construct_candidates(a, k, n):
  return [True, False]

def make_move(a, k, input):
  pass

def unmake_move(a, k, input):
  pass

finished = False

def backtrack(a, k, the_input):
  if is_solution(a, k, the_input):
    process_solution(a, k, the_input)
  else:
    k += 1
    for in construct_candidates(a, k, n):
      a[k - 1] = i
      make_move(a, k, the_input)
      backtrack(a, k, the_input)
      unmake_move(a, k, the_input)
      if finished:
        return



# Back tracking SUDOKU

# List of blocks in the same row or same column or the 3 *3 block
adjacent = {}

for i in ['012', '345', '678']:
  for j in ['012', '345', '678']:
    square = set([(int(a), int(b)) for a in i for b in j])
    for k in square:
      adjacent[k] = square 
      adjacent[k] = adjacent[k] - set([k])

for i in range(9):
  for j in range(9):
    row = set([(int(i), k) for k in range(9) if k != j])
    column = set([(k, int(j)) for k in range(9)  if k != i])
    adjacent[(i,j)] = adjacent[(i,j)] | row | column


def is_solution(the_input):
  return  all([j >=1 and j <=9 for i in the_input for j in i])

def process_solution(the_input):
  print the_input

def construct_candidates(the_input):
  row, col = find_first_missing(the_input)
  for n in range(1,10):
    if will_fit(the_input, row, col, n):
      yield (row, col, n)

import operator
def find_first_missing(the_input):
  values = {}
  for row in range(9):
    for col in range(9):
      if the_input[row][col] > 0:
        continue
      val = 0 
      for i, j in adjacent[(row, col)]:
        if the_input[i][j] > 0:
          val += 1
      values[(row, col)] = val
  return sorted(values.items(), key=operator.itemgetter(1), reverse = True)[0][0]



def find_first_missing(the_input):
  for i in range(9):
    for j in range(9):
      if the_input[i][j] == 0:
        return i, j


def will_fit(the_input, row, col, n):
  for i, j in adjacent[(row, col)]:
    if the_input[i][j] == n:
      return False
  return True

def make_move(the_input, pos):
  i, j, n = pos
  the_input[i][j]  = n

def unmake_move(the_input, pos):
  i, j, n = pos
  the_input[i][j]  = 0


def backtrack(the_input):
  global finished
  if is_solution(the_input):
    process_solution(the_input)
    print  'Setting finished %s' % the_input
    finished = True
  else:
    for i in construct_candidates(the_input):
      print 'Trying to fit  %s into (%s, %s) with current board %s' % (i[2], i[0], i[1], the_input)
      make_move(the_input, i)
      backtrack(the_input)
      if finished:
        return
      unmake_move(the_input, i)


def sudoku(board):
"""
pass the grid as a new line delimited string like 
sudoku('''200080300
060070084
030500209
000105408
000000000
402706000
301007040
720040060
004010003''')
"""
board = """003020600
900305001
001806400
008102900
700000008
006708200
002609500
800203009
005010300"""
rows = board.split('\n')
grid = []
for i_index, i in enumerate(rows):
  grid.append([])
  for j_index, j in enumerate(list(i.strip())):
    grid[i_index].append(int(j))
print backtrack(grid)


13


def lucky(n):
  pos = n
  if pos%2 == 0:
    return False
  pos = (n + 1) /2
  for i in range(3,n):
    print pos, i
    if i > pos:
      return True
    reminder = pos % i
    if reminder == 0:
      return False
    else:
      pos = pos  - (pos/i)
    

1800 3000 8282
1800 266 8282

SN  -> 320 10 11 50 94


AAAAAAAA
12 5   8



AAAAA
12  5

5 * 5 * 5 * 5 * 5 * 5
14

8 * 8 * 8

AAAAAA
7 

def max_no_of_as(n):
  print n
  if n <= 6:
    return n
  else:
    m = n - 2
    for i in range(2, n -3):
      cur = (n - i - 1 ) *  max_no_of_as(i)
      if cur > m:
        m = cur
    return cur


def repeat(a):
  for i in range(len(a)):
    print i, a[i], a[a[i]]
    if a[abs(a[i])] < 0:
      yield abs(a[i])
    else:
      a[abs(a[i])] *= -1
    print a


1 2 3 4 5 6 7 8 9
1 2 3 4   6 7 8 9
1 2 3   5 6 7 8 9
1 2 3 4 5   7 8 9

[1,2,3,4,6,7,8,9]

0 1  2  3 4 5 6 7 8
4
5
def missing(a, start = None, end = None):
  start = start or 0
  end = end or len(a) - 1
  print start, end
  if len(a[start:end + 1]) == 2:
    return a[start] + 1
  elif len(a[start:end + 1]) == 1:
    if a[start] < start + 1:
      return a[start] + 1
    else:
      return a[start] - 1
  else:
    mid = start  + (end - start)/2
    print 'mid' , mid +1, a[mid]
    if a[mid] == mid + 1:
      return missing(a, end - (end - start)/2 -1, end)      
    else:
      return missing(a, start,  start + (end - start)/2)


  
missing([1,2,3,4,6,7,8,9])
  
  
  



def missing(a):
  if a[(a[-1] - 1)/2] == (a[-1]  + 1)/2:


def bin_search(a, n, start = 0, end = None):
  time.sleep(0.5)
  start = start or 0
  end = end or len(a) - 1
  print start, end
  if len(a[start:end + 1]) == 1:
    if a[start] == n:
      return start
    else:
      return -1
  else:
    middle = start + (end - start)/2
    if a[middle] == n:
      return middle
    elif a[middle] > n:
      return bin_search(a, n, start, start + (end - start) /2)
    else:
      return bin_search(a, n, end - (end - start) /2, end)


bin_search(range(10), 5)




def gold_mine(a, i, j):
  if i < 0 or j < 0 or i >= len(a) or j >= len(a[0]):
    return 0
  if j == len(a[0]) -- 1:
    return a[i][j]
  else:
    return max(a[i][j] + gold_mine(a, i, j + 1), a[i][j] + gold_mine(a, i - 1, j + 1), a[i][j] + gold_mine(a, i + 1, j + 1))



def gold_mine_main(a):
  m = 0
  for i in range(len(a)):
    cur = gold_mine(a, i, 0)
    if cur > m:
      m = cur
  return m


gold_mine_main([[1, 3, 1, 5], [2, 2, 4, 1], [5, 0, 2, 3], [0, 6, 1, 2]])

gold_mine([[1, 3, 1, 5], [2, 2, 4, 1], [5, 0, 2, 3], [0, 6, 1, 2]], 2, 0)



def element_once(a):
  bit_count = [0] * 32
  for i in a:
    bit_count = [k[0] + int(k[1]) for k in zip(bit_count, list('{0:032b}'.format(i)))]
  for i in range(32):
    bit_count[i] = bit_count[i] % 3
  return int(''.join(map(str, bit_count)), 2)



def median_2_sorted(a, b):
  la = len(a)
  lb = len(b)
  lc = la + lb
  if lc % 2 == 0:
    median_elements = [(lc - 2)/2, lc /2]
  else:
    median_elements = [lc/2]
  i = 0
  j = 0
  median = []
  print median_elements
  for k in range(lc):
    if k > (lc + 1) /2 + 1:
      break
    if a[i] < b[j]:
      if k in median_elements:
        median.append(a[i])
      i += 1
    else:
      if k in median_elements:
        median.append(b[j])
      j += 1
  print median


median_2_sorted([1,3,5,11,17],  [9,10,11,13,14])


[-1,-2,-3,-4,1,2,3,4,5]

def shift(a):
  current_pos = -1
  next_pos = -1
  next_neg = -1
  for index,i in enumerate(a):
    if i > 0:
      next_pos = index
      break
  for index,i in enumerate(a):
    if i < 0:
      next_neg = index
  if next_pos == -1 or next_neg == -1:
    return a
  while True:
    if next_pos = current_pos + 1 and next_neg = current_pos + 1:
      current_pos = current_pos + 2
    elif next_pos = current_pos + 1:
      a[next_neg], a[next_pos + 1] = a[next_pos + 1],  a[next_neg]
      current_pos = current_pos + 2
    elif next_neg = current_pos + 1:



def next_element(a):
  n = [0] * len(a)
  s = []
  for index,i in enumerate(a):
    if not s:
      s.append(i)
      continue
    if i < s[-1]:
      s.append(i)
    else:
      n[index - 1] = i
      s.pop()
      j = 1
      while s[-1] < i:
        n[index - 1 - j] = i
        s.pop()
        if len(s) == 0:
          break
      s.append(i)
  print n


next_element([98, 23, 54, 12, 20, 7, 27])

def peak_element(a):
  i = 0

def alternate()

