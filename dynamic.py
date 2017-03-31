# Dynamic binomial coefficents

def binomial(n, k):
  a = []
  a.append([1, 0])
  for i in range(1, n):
    a.append([i + 1])
    for j in range(1, i + 1):
      a[i].append(a[i-1][j-1] + a[i-1][j])
    a[i].append(0)
  print a
  return a[n-1][k]


ab c
de f 

d(a, b, i, j) = 1 if b[j] == null, 
                1 if a[i] == null, 
                d(a, b, i - 1, j - 1) if a[i] = b[j]
                min(i - 1, j - 1 + 1, i - 1, j +  1, i, j + 1 + 1) 

# distance between 2 string
def distance(a, b, i, j):
  d = []
  x = max(len(a), len(b))
  la = len(a)
  lb = len(b)
  d.append(range(lb + 1))
  print d
  for i in range(la):
    d.append([i + 1] + [None] * lb)
    for j in range(lb):
      print i, j
      if i > la -1 :
        d[i + 1][j + 1] = d[i][j] + 1
      elif j > lb -1:
        d[i + 1][j + 1] = d[i][j] + 1
      else:
        match = d[i][j] if a[i] == b[i] else d[i][j] + 1
        d[i + 1][j + 1] = max(match, d[i][j + 1] + 1, d[i + 1][j] + 1)
  return d

cache = {}

def min_moves(a, b):
  return _min_moves(a, b, len(a) - 1, len(b) - 1)

# recursive
def _min_moves(a, b, i, j):
  print a, b, i, j
  if i == -1:
    return 1 + j if j >=0 else 0
  elif j == -1:
    return 1 + i if i >=0 else 0
  elif a[i] == b[j]:
    return _min_moves(a, b, i - 1, j -1)
  else:
    return min([_min_moves(a, b, i - 1, j -1) + 1,  _min_moves(a, b, i - 1, j) + 1, _min_moves(a, b, i , j -1) + 1])

# dynamic, caching
def _min_moves(a, b, i, j):
  if (i, j) in cache:
    return cache[(i, j)]
  elif i == -1:
    return 1 + j if j >=0 else 0
  elif j == -1:
    return 1 + i if i >=0 else 0
  elif a[i] == b[j]:
    x = _min_moves(a, b, i - 1, j -1)
    cache[(i, j)] = x
    return x
  else:
    x = min([_min_moves(a, b, i - 1, j -1) + 1,  _min_moves(a, b, i - 1, j) + 1, _min_moves(a, b, i , j -1) + 1])
    cache[(i, j)] = x
    return x


# dynamic, table
def _min_moves(a, b):
  t = []
  for i in range(len(a) + 1):
    t.append([1] * (len(b) + 1))
  t[0] = range(len(b) + 1)
  for index_i, i in enumerate(t):
    i[0] = index_i
  for i in range(len(a)):
    for j in range(len(b)):
      if a[i] == b[j]:
        t[i + 1][j + 1] = t[i][j]
      else:
        t[i + 1][j + 1] = min (t[i][j] + 1, t[i-1][j] + 1, t[i][j -1 ] + 1)
  print t




def short_pal(a):
  d = {}
  for index,i in enumerate(list(a)):
    if i in d:
      d[i].append(index)
    else:
      d[i] = [index]
  b = []

  for i in list(a):
    if len(d[i]) > 1:
      b.append(i)
  d1= {}
  for i in set(list(b)):
    d1[i] = d[i]

  for i in d1:

import operator
def longest_increasting_sub(a):
  b = []
  b.append((1,a[0]))
  for i in range(1,len(a)):
    m = (1, a[i])
    for j in range(i):
      print i, j, b
      if b[j][1] < a[i] and (b[j][0] + 1) > m[0]:
        m = (b[j][0] + 1, a[i])
    if m[0] == 0:
      m = (1, a[i])
    b.append(m)
  return sorted(b,  key = operator.itemgetter(1), reverse = True)[0][0]


# Possible ways of coin change
def rec_coin_change(coins, total):
  if total == 0:
    return 1
  if total < 0:
    return 0
  if len(coins) <=0 and total > 0:
    return 0
  return rec_coin_change(coins[:-1], total) + rec_coin_change(coins, total - coins[-1])

# Minimum coins required
def rec_min_coin_change(coins, total):
  if total == 0:
    return 0
  if not coins:
    return 1000000
  if total == 1:
    if 1 in coins:
      return 1
    else:
      return 1000000
  else:
    if coins[-1] <= total:
      return min( rec_min_coin_change(coins, total - coins[-1]) + 1, rec_min_coin_change(coins[:-1], total) )
    else:
      return rec_min_coin_change(coins[:-1], total)

# dynamic

def rec_min_coin_change(coins, total):
  mins = [0] * (total + 1)
  mins[0] = 0
  mins[1] = 1 if 1 in coins else 1000000
  for i in range(total + 1):
    if i in coins:
      mins[i] = 1
    else:
      cur_min = 10000000
      for j in coins:
        if j > total:
          continue
        else:
          cur_min = min(cur_min, mins[i - j] + 1)
      mins[i] = cur_min
  print mins

 
rec_coin_change([1,2,3], 4)


def coin_change(coins, n):
  m = len(coins)
  table = [[0 for x in range(m)] for x in range(n+1)]
  for i in range(m):
    table[0][i] = 1
  for i in range(1, n + 1):
    for j in range(m):
      table[i][j] = table[i - 1]
    for j in range(m):
      a = table[i]


# 10 4
# 2 5 3 6

# s(1, total) = 4
# s(2, toal) = s (1, total)  + s(2, 5)


# 1 2 3 4  1 2 3 4 5 
# 1 2 3 4  1 2 3 4 5


# Can an array be split into 2 parts which can be added to same value
def partition(a):
  if sum(a) % 2 == 1:
    return False
  else:
    return is_subset_sum(sorted(a), sum(a)/2)


def is_subset_sum(a, s):
  print a, s
  if len(a) == 0:
    if s == 0:
      return True
    else:
      return False
  elif len(a) == 1:
    if s == a[0]:
      return True
    else:
      return False
  elif a[-1] > s:
    return False
  else:
    return is_subset_sum(a[:-1], s) or is_subset_sum(a[:-1], s - a[-1])



def min_distance_between_string(a, b):
  if len(a) == 0:
    return len(b)
  elif len(b) == 0:
    return len(a)
  else:
    if a[-1] == b[-1]:
      return min_distance_between_string(a[:-1], b[:-1])
    else:
      return min(min_distance_between_string(a[:-1], b[:-1]) + 1, min_distance_between_string(a, b[:-1]) + 1, min_distance_between_string(a[:-1], b) + 1)



cat
car




def max_subrray_sum(a):
  t = []
  cur_max = a[0]
  for i in range(len(a)):
    t.append(a[i])
  for i in range(1, len(a)):
    if t[i -1] + a[i] > a[i] or (a[i] >= 0 and t[i -1] >= 0):
      t[i] = t[i -1] + a[i]
    else:
      t[i] = a[i]
    if t[i] > cur_max:
      cur_max = t[i]
  return cur_max






def total_sub_trees(a):
  if a <= 1:
    return 1
  else:
    s = 0
    for i in range(a):
      s += total_sub_trees(i) * total_sub_trees(a - i - 1)
    return s



def subset_sum(a, s):
  if len(a) == 1:
    if s == a[0]:
      return True
    else:
      return False
  elif a[-1] == s:
    return True
  else:
    if a[-1] <= s:
      return subset_sum(a[:-1], s) or subset_sum(a[:-1], s - a[-1])
    else:
      return subset_sum(a[:-1], s)




def subset_sum(a, s):
  t = []
  for i in range(len(a) + 1):
    t.append([False] * (s + 1))
  t[0][0] = True
  for i in(range(len(a) + 1)):
    t[i][0] = True
  for i in range(1, len(a) + 1):
    for j in range(1, s + 1):
      if a[i - 1] <= j:
        t[i][j] = t[i - 1][j] or a[i-1] == s or t[i-1][j - a[i - 1]]
      else:
        t[i][j] = t[i - 1][j] or a[i-1] == s
  return t
 

subset_sum([1,3,9,2], 5)


def paranthesis(a):
  total = 0
  for i in list(a.strip()):
    if i == '(':
      total += 1
    elif i == ')':
      total -= 1
    if total < 0 :
      return False
  return True if total == 0 else False


paranthesis('()()()')

cache = {}
def min_cuts(a, n):
  if n in cache:
    return cache[n]
  else:
    if n == 1:
      x = a[0]
    elif n == 2:
      x = max(a[0] * 2, a[1])
    else:
      x = max ( min_cuts(a, n - 1) + a[0], min_cuts(a, n - 2) + a[1], a[n - 1] )
    cache[n] = x
    return x




min_cuts([3,5,8,9,10,17,17,20],  8)




cache = {}
def short_palyndrome(a):
  if a in cache:
    return cache[a]
  else:
    print a
    if len(a) == 1:
      k =  a
    elif len(a) == 2:
      k =  a if a[0] == a[-1] else a[0]
    else:
      if a[0] == a[-1]:
        k =  a[0] + short_pal(a[1:-1]) + a[-1] 
      else:
        x = short_pal(a[:-1])
        y = short_pal(a[1:])
        if len(x) > len(y):
          k =  x
        else:
          k =  y
  cache[a] = k
  return k

def subset_sum(a, s):
  print a, s
  if sum(a) == s:
    return True
  else:
    possible = False
    for i in range(len(a)):
      if s <= a[i]:
        possible =  possible or a[i] == s or  subset_sum(a[:i] + a[i + 1:], s - a[i]) 
      else:
        possible = possible or subset_sum(a[:i] + a[i + 1:], s - a[i]) 
    return possible



def word_breaks(s, w):
  print w
  if w in s:
    return True
  else:
    present = False
    for i in range(len(w)):
      if w[:i + 1] in s:
        present = present or word_breaks(s, w[i + 1:])
    return present


word_breaks(['i', 'like', 'sam', 'sung', 'samsung', 'mobile', 'ice', 'cream', 'icecream', 'man', 'go', 'mango'], 'ilike')


0000111100000

def binary_index_0(a):
  print a
  if len(a) <= 2:
    return 0 if (len(a) == 0 or a[1] == 1) else 1
  before = []
  for i in range(len(a)):
    before.append(0)
  after = []
  for i in range(len(a)):
    after.append(0)
  around = []
  for i in range(1, len(a)):
    before[i] += before[i - 1] + 1 if a[i - 1] == 1 else 0
  print before
  for i in reversed(range(len(a) - 1)):
    after[i] += after[i + 1] + 1 if a[i + 1] == 1 else 0
  print after
  m = 0
  m_pos = 0
  for i in range(len(a)):
    around.append(before[i] + after[i])
    if around[i] > m:
      m_pos = i
      m = around[i]
  print around
  return m_pos

binary_index_0([0,0,1,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,0,1,1,1,1,0])



def knapsack(values, weights, w):
  t = []
  for i in range(len(weights) + 1):
    t.append([0] * (w + 1))
  print t
  for i in range(1, (len(weights) + 1)):
    for j in range(1, (w + 1)):
      if weights[i - 1] <= j:
        t[i][j] = max(t[i-1][j - weights[i - 1]] + values[i - 1],  t[i-1][j])
      else:
        t[i][j] = t[i-1][j]
  for i in t:
    print i


knapsack([60, 100, 120], [20, 20, 30], 50)
