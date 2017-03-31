import time
def lps(a):
  lps = [0] * len(a)
  lps[1] = 1 if a[0] == a[1] else 0
  for index, i in enumerate(a):
    if index <= 1:
      continue
    if a[lps[index - 1]] == i:
      lps[index] = lps[index - 1] + 1
    else:
      j = lps[index - 1] - 1
      while True:
        if j <= 0:
          lps[index] = 0
          break
        if a[lps[j]] == i:
           lps[index] = lps[j] + 1
           break
        else:
          j = lps[j] - 1
          print j
  return lps



def match(p, s):
  lpsa = lps(p)
  i = 0
  j = 0
  while True:
    print i, j
    if i >= len(s):
      break
    if s[i] == p[j]:
      i += 1
      j += 1
      if j == len(p) - 1:
        print 'pattern found between %s %s' % (i - j + 1 and i + 1)
        j = 0
    else:
      j = lpsa[j - 1]
      


lps('AAACAAAAAC')
match('AAACAAAAAC', 'XXYYAAACAAAAACTTTAAACAAAAAC')