def pat(s, d):
  l = len(s)
  for j in d:
    if len(j) != l:
      continue
    t = {}
    used = set([])
    match = True
    for index, i in enumerate(s):
      if i in t and j[index] != t[i]:
        match = False
        break
      elif i not in t and j[index] in used:
        match = False
        break
      elif i not in t:
        t[i] = j[index]
        used.add(j[index])
    if match:
      yield j

aaaaabbb

aabbbaa

baabaab

def pal(s):
  d = {}
  for i in list(s):
    if i in d:
      d[i] += 1
    else:
      d[i] = 1
  m = None
  l = []
  r = []
  for i in d:
    if not m and d[i] == 1:
      m = i
    elif d[i]%2 == 0:
      for j in range(d[i]/2):
        l.append(i)
    else:
      for j in range(d[i]/2):
        l.append(i)
      if not m:
        m = i
  return ''.join(l) + m + ''.join(reversed(l)) if m else ''.join(l)  + ''.join(reversed(l))


