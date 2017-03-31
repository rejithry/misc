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