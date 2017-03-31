# Total number of sub trees

def cat(n):
  if n == 1 or n == 0:
    return 1
  else:
    total = 0
    for i in range(1, n + 1):
      total += cat(i - 1) * cat(n - i)
    return total

# Longest common sub sequence [ substring can be disjoint) , XBCD and XD has lss of XD
def lc_subseq(m, n):
  if len(m) == 0:
    return 0
  elif len(n) == 0:
    return 0
  elif m[-1] == n[-1]:
      return lc_subseq(m[:-1], n[:-1]) + 1
  else:
    return max(lc_subseq(m[:-1], n), lc_subseq(m, n[:-1]))

# Longest common sub string, no disjoint sub strings, rule is go diagonally

def lc_substring(m, n):
  max_len = 0
  max_len_pos = (0,0)
  a = []
  for i in range(len(m)):
    a.append([0] * len(n))
  print a
  for i in range(len(m)):
    for j in range(len(n)):
      if m[i] == n[j]:
        if i > 0 and j > 0:
          a[i][j] = a[i - 1][j - 1] + 1
        else:
          a[i][j] = 1
        max_len = max_len if a[i][j] < max_len else a[i][j]
        max_len_pos = (i, j)
  max_substring = ''
  for i in a:
    print i
  print  max_len
  i, j = max_len_pos
  while True:
    if a[i][j] == 1:
      return ''.join(list(reversed(max_substring + m[i])))
    else:
      max_substring += m[i]
      i = i - 1
      j = j - 1



lc_substring('xxxxxxabcdeffffff', 'abcabcd')


def knapsack_recursive(weights, benefits, total):
  if total <= 0:
    return 0
  elif len(weights) == 0:
    return 0
  else:
    if weights[0] <= total:
      return max(knapsack_recursive(weights[1:], benefits[1:], total - weights[0]) + benefits[0], knapsack_recursive(weights[1:], benefits[1:], total)  )
    else:
      return knapsack_recursive(weights[1:], benefits[1:], total)


def knapsack_dynamic(weights, benefits, total):
  a = []
  for i in range(len(weights)):
    a.append([0] * (total + 1))
  for i in range(len(weights)):
    for j in range(1, total + 1):
      if weights[i] <= j:
        if i >= 1 and j - weights[i] >= 0:
          a[i][j] = max(a[i-1][j - weights[i]] + benefits[i],  a[i-1][j])
        elif j - weights[i] >= 0:
          a[i][j] = a[i-1][j - weights[i]] + benefits[i]
        elif i >= 1:
          a[i][j] = a[i-1][j]
        else:
          a[i][j] = benefits[i]
      elif i - 1 > 0:
        a[i][j] = a[i-1][j]
  return a[-1][-1]


knapsack_dynamic([2,2,4,5], [3,7,2,9], 10)

  if total <= 0:
    return 0
  elif len(weights) == 0:
    return 0
  else:
    if weights[0] <= total:
      return max(knapsack_recursive(weights[1:], benefits[1:], total - weights[0]) + benefits[0], knapsack_recursive(weights[1:], benefits[1:], total)  )
    else:
      return knapsack_recursive(weights[1:], benefits[1:], total)
  

knapsack_recursive([2,2,4,5], [3,7,2,9], 10)

  


