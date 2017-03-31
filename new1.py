def rotate(a):
  for i in range(len(a)):
    for j in range(i):
      a[i][j], a[j][i] = a[j][i], a[i][j]
  for i in range(len(a)):
    for j in range(len(a)/2):
      a[i][j], a[i][len(a) -1 -j] = a[i][len(a) -1 -j],  a[i][j]

[[1, 2, 3, 4],
[4, 5, 6, 7],
[8, 9, 10, 11],
[12, 13, 14, 15]]

1 4 8   12
2 5 9   13
3 6 10  14
4 7 11  15


12 8 4 1
13 9 5 2
14 10 6 3
15 11 7 4



import sys

n, t = map(int, sys.stdin.readline().strip().split(' '))

a = map(int, sys.stdin.readline().strip().split(' '))

def build_index(a):
    index_a = [0] * n
    for index, i in enumerate(a):
        index_a[i - 1] = index
    return index_a

    
def swap(pos, target, a, index_a):
    
            a[n - i], a[index_a[i-1]] = a[index_a[i-1]], a[n - i]
            index_a[a[n - i] - 1], index_a[i - 1] = index_a[i - 1], index_a[a[n - i] - 1]    
i = n
k = 0


index_a = build_index(a)

while True:
    if k == t:
        break
    k_before = k
    
    for i in range(n):
        if k == t:
            break
        if a[i] < n - i:
            swap (i, n - i, a, index_a)
            k += 1            

    min_out_of_order_element_pos = n

    for i in a:
      if a[i] < n - i:
        if i < min_out_of_order_element_pos:
          min_out_of_order_element_pos = i

    for i
    for i in reversed(range(1, n + 1)):
        print a,i, a[n-i]
        if k == t:
            break
        if i_is_small()
        if  a[n - i] < i:
            print index_a

            print 'index', index_a
            print 'a', a
            k += 1
    k_after = k
    if k_before == k_after:
        break

print ' '.join(map(str, a))



g = {5 : set([2,0]), 4 : set([0,1]), 2 : set([3]), 3 : set([1]), 1 : set([]), 0 : set([])}
g_in_degree = {2 : set([5]), 0 : set([5,4]), 1 :set([4,3]), 3 : set([2]), 5 : set([]), 4 : set([])}


def topo(g):
  def find_v_with_0(g, visited):
    g_in_degree = {}
    for i in g:
      g_in_degree[i] = set([])
      for j in g[i]:
        g_in_degree[j] = set([])
    for i in g:
      for j in g[i]:
        g_in_degree[j].add(i)    
    print g_in_degree    
    for i in g_in_degree:
      if not g_in_degree[i] and visited[i] == 0:
        return i
    return None
  def sort(g, s, v, visited):
    print 'processing' , v
    visited[v] = 1
    if not g[v]:
      s.append(v)
      return 
    else:
      for j in g[v]:
        if visited[j] == 0:
          sort(g, s, j, visited)
      s.append(v)
      return
  visited = {}
  for i in g:
    visited[i] = 0
  s = []
  #v = find_v_with_0(g, visited)
  #while g:
  for v in g:
    sort(g,s, v, visited)
  #v = find_v_with_0(g, visited)
  return s


def partition(k, start, end):
  if start == end:
    return start
  pivot = k[start]
  j = start
  for index in range(start + 1 , end + 1):
    if k[index] < pivot:
      k[j], k[j + 1], k[index] = k[index], k[j], k[j + 1]
      j += 1
  return  j


def kth_largest(a, left , right, k):
  pivot_pos = partition(a, left, right )
  while True:
    if pivot_pos == left + k - 1:
      return a[pivot_pos]
    elif  pivot_pos < k - 1:
      return kth_largest(a, pivot_pos + 1, right, k - (pivot_pos - left) - 1 )
    else:
      return kth_largest(a, left, pivot_pos - 1, k)
    time.sleep(1)

kth_largest([7,13,4,5,1,2], 0, 5, 4)


import sys

q = int(sys.stdin.readline().strip())
for i in range(q):
    m, n =map(int, sys.stdin.readline().strip().split(' '))
    cm = sorted(map(int, sys.stdin.readline().strip().split(' ')), reverse = True)
    cn = sorted(map(int, sys.stdin.readline().strip().split(' ')), reverse = True)
    cur_x_segments = 1
    cur_y_segments = 1
    lm = len(cm)
    ln = len(cn)
    M = max(lm, ln)
    i = 0
    j = 0
    s = 0
    print cm
    print cn
    for k in range(lm + ln):
        if i > lm - 1:
            s += cm[j] * cur_x_segments 
            j += 1
            cur_y_segments += 1
        elif j > ln - 1:
            s += cm[i] * cur_y_segments 
            i += 1
            cur_x_segments += 1            
        elif cm[i] > cn[j]:
            s += cm[i] * cur_y_segments 
            i += 1
            cur_x_segments += 1 
        elif cm[i] == cn[j]:
            if (lm - cur_x_segments) > (ln - cur_y_segments) :
                s += cm[i] * cur_y_segments 
                i += 1
                cur_x_segments += 1 
            else:
                s += cm[j] * cur_x_segments 
                j += 1
                cur_y_segments += 1 
                
        else:
            s += cm[j] * cur_x_segments 
            j += 1
            cur_y_segments += 1
        print s
    
    
def min_coins(a, v):
  mins = []
  for i in sorted(a, reverse =True):
    print mins
    if len(mins) == 0:
      if i <= v:
        if v % i == 0:
          return [i] * (v/i)
        else:
          mins.append(([i] * v/i, v % i))
    else:
      for index, j in enumerate(mins):
        if  i <= j[1]:
          if j[1] % i == 0:
            return j[0] + [i] * (j[1]/i)
          else:
            mins[index] = (j[0] + [i] * (j[1]/i), j[1] - j[1] % i)
      if i <= v:
        mins.append(([i] * v/i, v % i))



def gcd(a, b):
  print a, b
  if a == 0:
    return b
  elif b == 0:
    return a
  else:
    r = gcd(a - b, b) if a > b else gcd(a, b - a)
    return r



def gcd1(a, b):
  print a, b
  if b == 0:
    return a
  else:
    return gcd1(b, a % b)


k[0], k[1], k[1] = 4, 3, 3