def matrix(a):
  m = len(a)
  n = len(a[0])
  j = 0
  i = m - 1
  total = 0
  while j < n -1:
    print i, j, total
    if i < 0:
      break
    if j > n - 1:
      break
    if a[i][j] == 0:
      total += i + 1
      j += 1
      i += 1
    else:
      i -= 1
  return total


a = [[0, 0, 1], [0,1,1], [1,1,1]]
a = [[1] * 4] * 4
matrix(a)


def max_prefix(a):
  m = []
  for i in range(len(a)):
    m.append(prefix(a, i))
  print m


def prefix(a, i):
  if i == len(a) - 1:
    return 0
  else:
    return prefix(a, i + 1) + 1 if a[i + 1] > a[i] else prefix(a, i + 1) 



def ramanujan(n):
  sums = {}
  for i in range(1, n + 1):
    i_cube = i * i * i
    if i_cube > n:
      break
    for j in range(i + 1, n + 1):
      j_cube = j * j * j
      s = j_cube + i_cube 
      if s > n:
        break
      if s in sums:
        sums[s] += 1
      else:
        sums[s] = 1
  for i in sums:
    if sums[i] >= 2:
      print i


ramanujan(2000)



def print_non_overlapping(s):
  _print_non_overlapping(s, '')


def _print_non_overlapping(s, prefix):
  print prefix + '(' + s + ')'
  for i in range(len(s) - 1):
    new_prefix = prefix + '(' + s[:i + 1] + ')'
    _print_non_overlapping(s[i + 1:], new_prefix)
