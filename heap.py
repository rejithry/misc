
def left_i(i):
  return  2 * i + 1

def right_i(i):
  return (2 * i) + 2

def parent_i(i):
  return (i/2) 

def max(a, left, right)

def heapify(a, i):
  left = left_i(i)
  right = right_i(i)
  if left < len(a) and a[left] > a[i]:
    if right < len(a) and a[right] > a[left]:
      a[right] , a[i] = a[i], a[right]
      heapify(a, right)
    else:
      a[left] , a[i] = a[i], a[left]
      heapify(a, left)
  elif right < len(a) and a[right] > a[i]:
    a[right] , a[i] = a[i], a[right]
    heapify(a, right)    

def get_max(a):
  a[0], a[-1] = a[-1], a[0]
  r =  a.pop()
  heapify(a, 0)
  return r

def heapsort(a):
  heapify_list(a)
  while a:
    print get_max(a)

def heapify_list(a):
  for i in reversed(range(len(a) / 2)):
    heapify(a, i)
  return a

def insert(a, i):
  a.append(i)
  cur = len(a) - 1
  while cur > 0:
    parent = parent_i(cur)
    print cur, parent
    if a[cur] > a[parent]:
      a[cur], a[parent] = a[parent], a[cur]
      cur = parent
    else:
      break


heapsort([1,4,5,7,2,10,9,3])

a = heapify_list([1,4,5,7,2,10,9,3])
insert(a, 11)



def sort(a):
  l = len(a)
  if l <= 1:
    return a
  else:
    return merge(sort(a[:l/2]), sort(a[l/2:]))


def merge(a, b):
  la = len(a)
  lb = len(b)
  lc = la + lb
  i = 0
  j = 0
  c = []
  for k in range(lc):
    if i == la:
      c.append(b[j])
      j += 1
    elif j == lb:
      c.append(a[i])
      i += 1      
    elif a[i] < b[j]:
      c.append(a[i])
      i += 1
    else:
      c.append(b[j])
      j += 1
  return c


def quick(a, start = None, end = None):
  start = start or 0
  if end == None:
    end = len(a) - 1
  if len(a[start:end]) == 1 or len(a[start:end]) == 0:
    return a[start:end]
  pivot = a[start]
  pivot_pos = start
  for j in range(start + 1, end):
    if a[j] < pivot:
      a[pivot_pos + 1], a[j] = a[j], a[pivot_pos + 1]
      a[pivot_pos], a[pivot_pos + 1] = a[pivot_pos + 1], a[pivot_pos]
      pivot_pos += 1
  return quick(a, start, pivot_pos) + [a[pivot_pos]] + quick(a, pivot_pos + 1, end)


quick([1,4,5,7,2,10,9,3])



def counting_sort(a, k):
  b = [0] * k
  c = [0] * len(a)
  for i in a:
    b[i] += 1
  s = -1
  for i in range(k):
    s += b[i]
    b[i] = s
  print b
  for i in reversed(range(len(a))):
    c[b[a[i]]] = a[i]
    b[a[i]] -= 1
    print c, b
  print c

counting_sort([0,1,0,5,4,3,4,5,5,5,5,5,5,7], 8)


def _radix(a, d):
  b = [0] * 10
  c = [0] * len(a)
  for i in a:
    b[int(str(i)[d])] += 1
  s = -1
  for i in range(10):
    s += b[i]
    b[i] = s
  for i in reversed(range(len(a))):
    c[b[int(str(a[i])[d])]] = a[i]
    b[int(str(a[i])[d])] -= 1
  return  c


def radix(a, k):
  for i in reversed(range(k)):
    a = _radix(a, i)
  print a

radix([324, 123, 560, 797, 893, 452, 678], 3)


def inversions(a, total = 0):
  if len(a) == 1:
    return (a, 0)
  else:
    total_x = inversions(a[:len(a)/2])
    total_y = inversions(a[len(a)/2:])
    return merge(total_x[0], total_y[0], total_x[1] + total_y[1])




def merge(a, b, total_inversion):
  print total_inversion
  la = len(a)
  lb = len(b)
  lc = la + lb
  i = 0
  j = 0
  c = []
  for k in range(lc):
    if i == la:
      c.append(b[j])
      j += 1
    elif j == lb:
      c.append(a[i])
      i += 1      
    elif a[i] < b[j]:
      c.append(a[i])
      i += 1
    else:
      c.append(b[j])
      total_inversion += (la - i)
      j += 1
  return (c, total_inversion)



inversions([4,2,3,5,1])
