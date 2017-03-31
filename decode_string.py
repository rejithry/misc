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
