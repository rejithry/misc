

def bfs(g, s):
  vertices = {}
  for i in g:
    vertices[i] = (-1, 'w', None, None) # Distance, Color, Pred, Path
  stack = [s]
  vertices[s] = (0, 'g', None, [s])
  while stack:
    v = stack.pop()
    vertices[v] = (vertices[v][0], 'b', vertices[v][2], vertices[v][3])
    for i in g[v]:
      if vertices[i][1] not in ['g', 'b']:
        stack.append(i)
        vertices[i] = (vertices[v][0] + 1, 'b', v, vertices[v][3] + [i])
  return vertices


def print_path(vertices, s, v):
  if v == s:
    print v
  elif not vertices[v][2]:
    print 'no path'
  else:
    print_path(vertices, s, vertices[v][2])
    print v

g = {1 : set([2,3,4]), 2 : set([3,4]), 3 : set([5]), 4 : set([5]), 5  : set([]), 6 :set([7]), 7 : set([6])}

bfs(g, 1)


def bfs_tree(g, s):
  t = {}
  vertices = {}
  for i in g:
    vertices[i] = (-1, 'w', None, None) # Distance, Color, Pred, Path
  stack = [s]
  vertices[s] = (0, 'g', None, [s])
  while stack:
    v = stack.pop()
    t[v] = set([])
    vertices[v] = (vertices[v][0], 'b', vertices[v][2], vertices[v][3])
    for i in g[v]:
      if vertices[i][1] not in ['g', 'b']:
        stack.append(i)
        t[v].add(i)
        vertices[i] = (vertices[v][0] + 1, 'b', v, vertices[v][3] + [i])
  return t

bfs_tree(g, 1)




#### DFS

l = []

def dfs(g):
  vertices = {}
  tree_edges = []
  fore_edges = []
  back_edges = []
  cross_edges = []
  for i in g:
    vertices[i] = (-1, -1, 'w', None) # Time1, Time2, Color, Pred
  time = 0
  #for i in ['s', 'z', 'y','x','w', 't', 'v', 'u']:
  for i in g:
    if vertices[i][2] == 'w':
      time = dfs_visit(g, i, vertices, time, tree_edges, fore_edges, back_edges, cross_edges)
  return (vertices, tree_edges, fore_edges, back_edges, cross_edges)

def dfs_visit(G, u, vertices, time, tree_edges = [], fore_edges = [], back_edges = [], cross_edges = []):
  time += 1
  vertices[u] = (time, vertices[u][0], 'g', vertices[u][3])
  for i in G[u]:
    if vertices[i][2] == 'w':
      vertices[i] = (vertices[i][0], vertices[i][1], vertices[i][2], u)
      time = dfs_visit(G, i, vertices, time, tree_edges, fore_edges, back_edges, cross_edges)
      tree_edges.append((u, i))
    elif vertices[i][2] == 'g':
      back_edges.append((u, i))
    elif vertices[i][2] == 'b' and vertices[u][0] > vertices[i][0]:
      cross_edges.append((u, i))
    else:
      fore_edges.append((u, i))
  time += 1
  vertices[u] = (vertices[u][0], time, 'b', vertices[u][3])
  l.insert(0, u)
  return time


g = {1 : set([2,3,4]), 2 : set([3,4]), 3 : set([5]), 4 : set([5]), 5  : set([]), 6 :set([7]), 7 : set([6])}

dfs(g)

def dfs(g):
  visited = []
  for i in g:
    if i not in visited:
      visited = visit(g, i, visited)
  print visited

def visit(g, j, visited):
  visited.append(j)
  for i in g[j]:
    if not i in visited:
      visited.append(i)
      visited = visit(g, i, visited)
  return visited

topo
tree_edges = []
fore_edges = []
back_edges = []
cross_edges = []

dfs(g)

{1: (1, 10, 'b', None), 2: (2, 9, 'b', 1), 3: (3, 6, 'b', 2), 4: (7, 8, 'b', 2), 5: (4, 5, 'b', 3), 6: (11, 14, 'b', None), 7: (12, 13, 'b', 6)}
import collections
g = {'y' : ['x']  ,'x' : ['z'] , 'z' : ['y', 'w'], 'w' : ['x'], 's' : ['z', 'w'], 'v' : ['w', 's'], 't' : ['v', 'u'], 'u' : ['v', 't'] }

# No back edge
g = {'y' : ['x']  ,'x' : [] , 'z' : ['y', 'w'], 'w' : ['x'], 's' : ['z', 'w'], 'v' : ['w', 's'], 't' : ['v', 'u'], 'u' : ['v'] }
g = collections.OrderedDict

1 2 3 4 5 6 7 8 9 10

g = { 'a' : ['b'], 'b' : ['e', 'f', 'c'],  'e' : ['a', 'f'] , 'f' : ['g'], 'g' : ['f', 'h'],  'c' : ['d', 'g'], 'd' : ['c', 'h'], 'h' : []}

def find_prime(g):
  g1 = {}
  for i in g:
    if not i in g1:
      g1[i] = []
    for j in g[i]:
      if not j in g1:
        g1[j] = [i]
      else:
        g1[j].append(i)
  return g1


from operator itemgetter


def scc(g):        
  vertices = dfs(g)[0]
  vertices2 = {}
  for i in g:
    vertices2[i] = (-1, -1, 'w', None) # Time1, Time2, Color, Pred
  time = 0
  g1 = find_prime(g)
  for i in [j[0] for j in sorted(vertices.items(), key = lambda x : x[1][1], reverse = True)]:
    tree_edges = []
    fore_edges = []
    back_edges = []
    cross_edges = []
    time_before = time
    if vertices2[i][2] == 'w':
      time = dfs_visit(g1, i, vertices2, time, tree_edges, fore_edges, back_edges, cross_edges)
    time_after = time
    if time_before == time_after - 2:
      print set([i])
    elif tree_edges:
      print set(reduce(lambda x,y : x + y, tree_edges))
  return vertices2

from heapq import heappush, heappop
from bitarray import bitarray
def prims(g, r):
  # Keep the heap of lowests
  min_vertext_heap = []
  # Map vertices to predecessor, and min_distance from mst when added
  vertext_map = {}
  # Map of vertices to 0 indices
  vertext_pos_map = {}
  # Membership byte array to hold if a vertex is already processed.
  membership = bitarray([True]) * len(g)
  # Mst edges
  mst = []
  position = 0
  # Push first element to heap with 0 distance, 
  heappush(min_vertext_heap , (0, r))
  # Assing vertext map with infinity for all vertex except the first, and set pred to None
  # Also set the vertex pos map
  for i in g:
    if i == r:
      vertext_map[i] = (0, None)
    else:
      vertext_map[i] = (10000000, None)
    vertext_pos_map[i] = position
    position += 1
  # While the heap is not empty
  while  min_vertext_heap:
    # Get min
    ver = heappop(min_vertext_heap)
    u = ver[1]
    # If the vertex is already processed skip
    if not membership[vertext_pos_map[u]]:
      continue
    # Add the min edge to MST, and mark it as processed.
    mst.append((vertext_map[u][1], u))
    print (vertext_map[u][1], u)
    membership[vertext_pos_map[u]] = False
    # All all vertices adjacent, which are not processed to heap, if the length is greater than the current distances.
    for i in g[u]:
      if membership[vertext_pos_map[i]] and g[u][i] < vertext_map[i][0]:
        vertext_map[i] = (g[u][i], u)
        heappush(min_vertext_heap , (g[u][i], i))
  return mst



A = [ [0,  4,  0,  0,  0,  0,   0,  8,  0],
      [4,  0,  8,  0,  0,  0,   0, 11,  0],
      [0,  8,  0,  7,  0,  4,   0,  0,  2],
      [0,  0,  7,  0,  9, 14,   0,  0,  0],
      [0,  0,  0,  9,  0, 10,   0,  0,  0],
      [0,  0,  4, 14, 10,  0,   2,  0,  0],
      [0,  0,  0,  0,  0,  2,   0,  1,  6],
      [8, 11,  0,  0,  0,  0,   1,  0,  7],
      [0,  0,  2,  0,  0,  0,   6,  7,  0]]


def to_adj_list(a):
  g = {}
  for i in range(len(a)):
    g[i] = {}
    for j in range(len(a)):
      if a[i][j] != 0:
        g[i][j] = a[i][j]
  return g

g = to_adj_list(A)

print g




def rec(a, k):
  #print a, k
  n = len(a)
  if k == 1:
    return sum(a)
  elif len(a) == 1:
    return a[0]
  elif len(a) == 0:
    return 0
  else:
    min_group = max ( rec(a[:n - 1], k -1), sum(a[n - 1:]))
    for i in reversed(range(n - 1)):
      cur = max ( rec(a[:i], k -1), sum(a[i:]))
      print min_group, cur, rec(a[:i], k -1), sum(a[i:]), i
      if cur < min_group:
        cur = min_group
    return min_group  

rec([5,4,1,12, 13], 3)
    
rec([100,200,300,400,500,600,700,800,900], 3)


def one_step_away(a, b):
  la = len(a)
  lb = len(b)
  if abs(la -lb) == 0:
    diffs = 0
    for i in range(la):
      if a[i] != b[i]:
        diffs += 1
        if diffs > 1:
          return False
    return True
  elif abs(la -lb) == 1:
    diffs = 0
    if lb > la:
      b, a = a, b
    j = 0
    for i in range(la):
      if a[i] != b[j]:
        diffs += 1
        if diffs > 1:
          return False
      else:
        j += 1
    return True
  else:
    return False


one_step_away('cat', 'cot')


def min_path(dictionary, a, b):
  g = {}
  for i in dictionary:
    g[i] = set([])
  for i in dictionary:
    for j in dictionary:
      if i != j:
        if one_step_away(i, j):
          g[i].add(j)
  print g
  vertices = bfs(g, a, b)
  print vertices
  if vertices == -1:
    print 'No path'
  else:
    print_path(vertices, a, b)


def bfs(g, s, t):
  vertices = {}
  for i in g:
    vertices[i] = (-1, 'w', None, None) # Distance, Color, Pred, Path
  stack = [s]
  vertices[s] = (0, 'g', None, [s])
  while stack:
    v = stack.pop()
    vertices[v] = (vertices[v][0], 'b', vertices[v][2], vertices[v][3])
    if  v == t:
      return vertices
    for i in g[v]:
      if vertices[i][1] not in ['g', 'b']:
        stack.append(i)
        vertices[i] = (vertices[v][0] + 1, 'b', v, vertices[v][3] + [i])
  return -1


def print_path(vertices, s, v):
  if v == s:
    print v
  elif not vertices[v][2]:
    print 'no path'
  else:
    print_path(vertices, s, vertices[v][2])
    print v


min_path(['BCCI', 'AICC', 'ICC', 'CCI', 'MCC', 'MCA', 'ACC'], 'AICC', 'MCA')



