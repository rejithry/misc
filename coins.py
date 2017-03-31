a = [10,15,55]
cache = set(a)
def coins(n):
    if n in cache:
      return n
    else:
      if n - 10 in cache or n -15 in cache or n - 55 in cache:
        cache.add(n)
        return n
      else:
        return None



def combination():
  for i in range(1, 1001):
    if coins(i):
      print i




def shortest_substring(string, alphabet):
  min_size = len(string)
  min_string = string
  left = 0
  right = 1
  cur_alphabet = {}
  for i in alphabet:
    cur_alphabet[i] = 0
  cur_alphabet[string[left]] += 1
  cur_alphabet[string[right]] += 1
  cur_string_size = len([i for i in cur_alphabet if cur_alphabet[i] > 0])
  while True:
    import time
    time.sleep(0.1)
    print cur_alphabet, string[left:right + 1], cur_string_size
    if cur_string_size == len(alphabet):
      left += 1
      cur_alphabet[string[left - 1]] -= 1
    else:
      right += 1
      if right > len(string) - 1:
        break
      cur_alphabet[string[right]] += 1
    cur_string_size = len([i for i in cur_alphabet if cur_alphabet[i] > 0])
    if cur_string_size == len(alphabet) and right - left < min_size:
      min_size = right - left
      min_string = string[left:right + 1]
  return min_string


shortest_substring('aabbccba', 'abc')



def largest_integer_to_right(a):
  pivot = -1
  first_non_zero_found = False
  for index, i in enumerate(a):
    if not first_non_zero_found:
      if i != 0:
        print pivot
        first_non_zero_found = True
      else:
        pivot += 1
    else:
      if i == 0:
        pivot += 1
        a[index], a[pivot] = a[pivot], a[index]
  print a

largest_integer_to_right([0,0,0,3,4,0,5,0,0,0,0])




def decode_string(s):
  if len(s) == 0 or s == '':
    return ''
  if len(s) == 1:
    return s[0]
  else:
    if s[0].isdigit():
      j = 1
      while j < len(s):
        if not s[j].isdigit():
          break
        j += 1
      k = 1
      l = j + 1
      while l < len(s):
        if s[l] == ']':
          k -= 1
        elif s[l] == '[':
          k += 1
        if k == 0:
          break
        l += 1
      return (int(s[:j]) * decode_string(s[j + 1:l])) + decode_string(s[l + 1:])
    else:
      return s[0] + decode_string(s[1:])


decode_string('3[a2[bd]g4[ef]h]')



def decode_string(s):
  stack = []
  string_so_far = ''
  multiplier = None
  i = 0
  for i in range(len(s)):
    print stack
    if s[i] == '[':
      stack.append((string_so_far, multiplier))
      string_so_far = ''
    elif s[i] == ']':
      t, m = stack.pop()
      string_so_far =  t + (m * string_so_far)
    elif not s[i].isdigit():
      string_so_far += s[i]
    else:
      multiplier = int(s[i])
  print string_so_far


decode_string('2[c3[ab]]')
decode_string('3[a2[bd]g4[ef]h]')

import random
def rand2():
  return random.randint(0, 1)

def rand3():
  x = rand2()
  y = rand2()
  while x == 1 and y == 0:
    x = rand2()
    y = rand2()
  return x + y



import random
def rand5():
  return random.randint(1, 5)

def rand7():
  z = rand5() * 5 + rand5()  - 5 
  while z > 21:
    z = rand5() * 5 + rand5()  - 5 
  return (z % 7) + 1
