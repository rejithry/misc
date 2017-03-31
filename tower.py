def tower(n):
  if n == 1 :
    return 1
  elif n == 2:
    return 3
  else:
    return 2 * tower(n - 1) + 1



def _tower_moves(a, from, to, with):
  if n == 1 :
    return []
  elif n == 2:
    return 3
  else:
    return 2 * tower(n - 1) + 1


g = { 1 : { 2 : 2, 3: 10}, 3 : { 4 : 1, 5 : 22}, 2 : {4 : 2}}

def find_min_node_and_update(v):
  s = sorted( [(i, v[i][0]) for i in v if not v[i][1] and v[i][0] != -1], key = lambda x : x[1] )
  if len(s) == 0:
    return None, None
  else:
    v[s[0][0]] = (s[0][1], True)
    return s[0][0], s[0][1]

import operator

def dijkstra(g, s, goal):
  vertices = {}
  for i in g:
    vertices[i] = (-1, False)
    for j in g[i]:
      vertices[j] = (-1, False)
  vertices[s] = (0, True)
  current = s
  distance_to_current = 0
  while True:
    print current
    if current in g:
      for i in g[current]:
        if not vertices[i][1]:
          new_distance = distance_to_current + g[current][i]
          if new_distance < vertices[i][0] or vertices[i][0] == -1:
            vertices[i] =  (new_distance, False)
      print vertices, current
    current, distance_to_current = find_min_node_and_update(vertices)
    print current, distance_to_current
    if not current:
      return
    if current == goal:
      return distance_to_current

dijkstra(g, 1, 4)


def shortest_dist_node(dist):
    best_node = 'undefined'
    best_value = 1000000
    for v in dist:
        if dist[v] < best_value:
            (best_node, best_value) = (v, dist[v])
    return best_node

def dijkstra(G,v):
    dist_so_far = {}
    dist_so_far[v] = 0
    final_dist = {}
    while len(final_dist) < len(G):
        w = shortest_dist_node(dist_so_far)
        # lock it down!
        final_dist[w] = dist_so_far[w]
        del dist_so_far[w]
        for x in G[w]:
            if x not in final_dist:
                if x not in dist_so_far:
                    dist_so_far[x] = final_dist[w] + G[w][x]
                elif final_dist[w] + G[w][x] < dist_so_far[x]:
                    dist_so_far[x] = final_dist[w] + G[w][x]
    return final_dist


def dijkstra(G,v):
    dist_so_far = {}
    dist_so_far[v] = 0
    final_dist = {}
    while len(final_dist) < len(G):
        w = shortest_dist_node(dist_so_far)
        # lock it down!
        del dist_so_far[w]
        for x in G[w]:
            if x not in final_dist:
                if x not in dist_so_far:
                    dist_so_far[x] = final_dist[w] + G[w][x]
                elif final_dist[w] + G[w][x] < dist_so_far[x]:
                    dist_so_far[x] = final_dist[w] + G[w][x]
    return final_dist

#Haep implmentation
from heapq import heappush, heappop


def dijkstra(G,v):
    dist_so_far = []
    heappush(dist_so_far , (0, v))
    final_dist = {}
    while len(final_dist) < len(G):
        best_node = heappop(dist_so_far)
        w = best_node[1]
        # lock it down!
        final_dist[w] = best_node[0]
        for x in G[w]:
            if x not in final_dist:
                heappush(dist_so_far, (final_dist[w] + G[w][x], x ))
    return final_dist


# Dijkstra with path
def dijkstra(G,v):
    dist_so_far = []
    print 'here'
    heappush(dist_so_far , (0, (v, None)))
    print 'here'
    final_dist = {}
    while len(final_dist) < len(G):
        best_node = heappop(dist_so_far)
        w = best_node[1][0]
        pred = best_node[1][1]
        # lock it down!
        final_dist[w] = (best_node[0], pred)
        for x in G[w]:
            if x not in final_dist:
                print 'here'
                print dist_so_far
                heappush(dist_so_far, (final_dist[w][0] + G[w][x], (x , w)) )
    return final_dist


def print_path(final_dist, s, v):
  if not v:
    return
  else:
    print_path(final_dist, s, final_dist[v][1])
    print v

g = { 1 : { 2 : 2, 3: 10}, 3 : { 4 : 1, 5 : 22}, 2 : {4 : 2}, 4 : {}, 5 : {}}

import sys

q = int(sys.stdin.readline().strip())
for h in range(q):
  V, E = map (int, sys.stdin.readline().strip().split(' '))

  g = {}
  for i in range(E):
    s, t, w = map (int, sys.stdin.readline().strip().split(' '))
    if s in g:
      if t in g[s]:
        if w < g[s][t]:
          g[s][t] = w
      else:
        g[s][t] = w
    else:
      g[s] = {t : w}
    if t in g:
      if s in g[t]:
        if w < g[t][s]:
          g[t][s] = w
      else:
        g[t][s] = w
    else:
      g[t] = {s : w}
  source = int(sys.stdin.readline().strip())
  print g, source
  print [i[1] for i in sorted(dijkstra(g,source).items()[1:], key =  lambda x : x[1])]
