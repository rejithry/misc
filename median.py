def partition(a):
  pivot = a[0]
  pivot_pos = 0
  first_large_element_found = False
  for i in range(1, len(a)):
    if not first_large_element_found:
      if a[i] <= pivot:
        pivot_pos += 1
      else:
        first_large_element_found = True
    else:
      if a[i] <= pivot:
        pivot_pos += 1
        a[i], a[pivot_pos] = a[pivot_pos], a[i]
  a[0], a[pivot_pos] = a[pivot_pos], a[0]
  return pivot_pos


def kth_elment(a, k):
  pivot_pos = partition(a)
  if len(a) == 2:
    if k == 0:
      return a[0]
    elif k == 1:
      return a[1]
  elif len(a) == 1:
    return a[0]
  elif pivot_pos == k:
    return a[pivot_pos]
  elif pivot_pos < k:
    return kth_elment(a[pivot_pos + 1:], k - pivot_pos - 1)
  else:
    return kth_elment(a[:pivot_pos], k)


def median(a):
  return kth_elment(a, len(a)/2)

import random

a = [random.randint(1, 200) for i in range(100)]
print 'median', median(a)
print 'median', sorted(a)[50]


