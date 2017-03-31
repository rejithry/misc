SELECT ga.the_date AS the_date
    ,ga.app_acronym AS app_acronym
	,ga.app_version AS app_version
	,ga.app_description AS app_description
	,ga.bundle_id AS bundle_id
	,ga.distribution_platform AS distribution_platform
	,ga.first_boot_date AS first_boot_date
	,ga.first_country AS first_country
	,ga.is_jailbroken AS is_jailbroken
	,ga.context AS context
	,ga.LEVEL AS LEVEL
	,ga.action AS action
	,ga.type AS type
	,ga.location AS location
	,(CASE WHEN a.num_payments >= 1 THEN 1 ELSE 0 END) AS payer_status
	,a.device_model AS device_model
	,a.device_machine AS device_machine
	,a.network AS network
	,a.campaign_name AS campaign_name
	,a.campaign_id AS campaign_id
	,a.site_id AS site_id
	,a.creative AS creative
	,a.facebook_campaign_id AS facebook_campaign_id
	,a.facebook_campaign_name AS facebook_campaign_name
	,ui.age_in_app AS age_in_app
	,ui.universal_age AS age
	,count(DISTINCT ga.dgmindex, 1000000) AS num_users
	,sum(ga.num_actions) AS num_actions
FROM dmo_devices.dismo_game_action_summary_20160601 ga
LEFT JOIN each dmo_devices.dismo_activity_20160601 a ON ga.dgmindex = a.dgmindex
	AND ga.app_acronym = a.app_acronym
LEFT JOIN each dmo_devices.dismo_user_age_gender_info ui ON ga.dgmindex = ui.dgmindex
	AND ga.app_acronym = ui.app_acronym
	  group each by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26;

def permutations(head, tail= []):
    if len(head) == 0: print tail
    else:
        for i in range(len(head)):
            permutations(head[0:i] + head[i+1:], tail + [head[i]])ef

l = []
def comb(a,n):
	if n == 1:
		for i in a:
			yield [i]
	else:
		for i in range(len(a)):
			for j in comb(a[i + 1:], n - 1):
				yield [a[i]] + j


def perm(a,n):
	if n == 1:
		for i in a:
			yield [i]
	else:
		for i in range(len(a)):
			for j in perm( a[:i] + a[i + 1:], n - 1):
				yield [a[i]] + j

		0
		[1,2] 2

		[0] [		


class Node:
	def __init__(self, a, parent = None, dir = None):
		self.a = a
		self.left = None
		self.right = None
		self.distance = 0
		self.next = None
		if parent and dir == 'r' and parent.left:
			parent.left.next = self
		elif parent and dir == 'l' and parent.right:
			self.next = parent.right
	def add_right(self, r, distance):
		n = Node(r, self, 'r')
		n.distance = distance
		self.right = n
	def add_left(self, l, distance):
		n = Node(l, self, 'l')
		n.distance = distance
		self.left = n



class Tree:
  def __init__(self):
    self.root = None
    self.head = None
    self.cur = None
    self.linked_list = LinkedList()
  def add(self, e, root = None, distance = 0):
    root  = root or self.root
    if not self.root:
      self.root = Node(e)
      self.root.distance = distance
    elif e < root.a:
      if not root.left:
        root.add_left(e, distance -1)
      else:
        self.add(e, root.left, distance -1)
    else:
      if not root.right:
        root.add_right(e, distance + 1)
      else:
        self.add(e, root.right, distance + 1)
  def serialize(self, root = None,current = ''):
  	root = root or self.root
  	left_tree = self.serialize(root.left) if root.left else ''
  	right_tree = self.serialize(root.right) if root.left else ''
  	return left_tree + '<-' + self.root.a + '->' + self.right.a
  def traverse(self, root = None):
    root = root or self.root
    self.traverse(root.left) if root.left else None
    print str(root.a) + ':' + str(root.distance)
    self.traverse(root.right) if root.right else None
  def tr(self, root = None, levels = None):
    root  = root or self.root
    levels = levels or {}
    if not root:
      return levels
    print str(root.a) + ':' + str(root.distance)
    if not root.distance in levels:
      levels[root.distance] = [root.a]
    else:
      levels[root.distance].append(root.a)
    if root.right:
      self.tr(root.right, levels)
    if root.left:
      self.tr(root.left, levels)
    return levels
  def is_balanced(self, root = None):
    root = root or self.root
    if not root:
      return True
    else:
      l = self.height(root.left)
      r = self.height(root.right)
      print root,l,r
      if l == -1 or r == -1:
        return False
      elif abs(l - r) > 1:
        return False
      else:
        return self.is_balanced(root.left) and self.is_balanced(root.right)
  def height(self, root):
    if not root:
      return 0
    else:
      r = self.height(root.right)
      l = self.height(root.left)
      if l - r > 1:
        return -1
      else:
        return max(l, r) + 1
  def convert_to_linked_direct(self):
    if self.root:
      self._convert_to_linked_direct(self.root)
  def _convert_to_linked_direct(self, root):
    if root:
      if root.left:
        self._convert_to_linked_direct(root.left)
      self.linked_list.add(root.a)
      if root.right:
        self._convert_to_linked_direct(root.right)
  def convert_to_linked(self, root = None, cur = None):
    if not root:
      root = self.root
    self._convert_to_linked(self.root, None)
  def _convert_to_linked(self, root, prev):
    if not root:
      return null
    if root.left:
      self._convert_to_linked(root.left, prev)
    if not prev:
      self.head = root
    else:
      prev.right = root
      root.left = prev
    prev = root
    if root.right:
      self._convert_to_linked(root.right, root)

t = Tree()
t.add(5)
t.add(3)
t.add(7)
t.add(4)
t.add(2)
t.add(6)
t.add(8)
t.convert_to_linked_direct()
t.linked_list.traverse()



#Doubly linked list
class Node:
  def __init__(self, a):
    self.a = a
    self.right = None
    self.left = None

class LinkedList:
  def __init__(self):
    self.head = None
    self.cur = None
  def add(self, a):
    n = Node(a)
    if not self.head:
      self.head = n
      self.cur = self.head
    else:
      n = Node(a)
      self.cur.next = n
      n.prev = self.cur
      self.cur = n
    n.next = self.head 
    self.head.prev = n
  def traverse(self):
    cur = self.head
    while True:
      print cur.a
      cur = cur.next
      if cur.next == self.head:
        print cur.a
        break


t = Tree()
[t.add(i) for i in [6,3,4,5,9,2,7,1,0,8]]
d = t.tr()
s = sorted(t.tr())
[ [ j for j in d[i] ] for i in s] ]



class Heap:
	def __init__(self):
		self.root = None
	def add(self, e):
		if not self.root:
			self.root = Node(e)
		else self._add(e, root=self.root)
	def _add(e, root)
		elif e < root.a:
			if not root.left:
				root.add_left(e)
			else:
				self.add(e, root.left)
		else:
			if not root.right:
				root.add_right(e)
			else:
				self.add(e, root.right)
	def traverse(self, root = None):
		root = root or self.root
		self.traverse(root.left) if root.left else None
		print root.a
		self.traverse(root.right) if root.right else None



G = {1 : [2,3] , 2: [4, 1], 3 : [4, 1], 4 : [2,3]}





#BFS non-directional graph

def bfs(g, start, end):
	s  = []
	path = []
	s.append(start)
	distance = {}
	distance[start] = 0
	while (s):
		visited_node = s.pop()
		path.append(visited_node) if not visited_node in path else None
		if visited_node == end:
			break
		[s.insert(0, n) for n in g[visited_node] if n not in path] if visited_node in g else None
		for n in g[visited_node]:
			if n not in path:				
				distance[n] = distance[visited_node] + 1 
	return (path, distance)

#Iterative dfs
def dfs(g, start, end):
	s = [start]
	print s
	path = []
	while s:
		t = s.pop()
		print t
		[path.append(t) if t not in path else None]
		if t == end:
			break
		[s.append(i) for i in g[t] if i not in path]
		print s
	return path


def bfs(g, start, end):
	s = [(start, [start])]
	while s:
		t, cur_path = s.pop()
		if t == end:
			yield cur_path
		[ s.insert(0, (i, cur_path + [i])) for i in g[t] if i not in cur_path]


def dfs(graph, start, visited = None):
	if not visited:
		visited = []
	visited.append(start) if not start in visited else None
	for next in [ i for i in graph[start]] :
		if i in visited:
			print 'Cycle'
			break
		dfs(graph, next, visited)
	return visited




#Iterative dfs paths
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

#Recursive dfs
def DFS(gr, s, path):
    if s in path: return False
    path.append(s)
    for each in gr[s]:
        if each not in path:
            DFS(gr, each, path)
    return path



def dfs_paths(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        yield path
    for next in graph[start] - set(path):
        for path in dfs_paths(graph, next, goal, path + [next]):
        	yield path




def grep(pattern):
	print "Looking for %s" % pattern
	while True:
	line = (yield)
		if pattern in line:
 	print line,


def print_args(func):
  def wrap(*args, **kwargs):
    print args
    return func(*args, **kwargs)
  return wrap

def print_args(n):
  def print_n(func):
    def wrap(*args, **kwargs):
      for i in range(n):
        print args
      return func(*args, **kwargs)
    return wrap
  return print_n


@print_args
def add(a, b):
    return a + b



def merge_sort(a):
	if len(a) < 2:
		return a 
	return merge( merge_sort(a[:(len(a) + 1)/2]), merge_sort(a[ (len(a)+1)/2 : ]) )



def merge(a, b):
	c = []
	j =0
	k = 0
	for i in range((len(a) + len(b))):
		if j == len(a) :
			return  c + b[k:]
		elif k == len(b):
			return  c + a[j:]
		elif a[j] < b[k]:
			c.append(a[j])
			j += 1
		else:
			c.append(b[k])
			k += 1
	return c


def quick_sort(a):
	if len(a) < 2:
		return a
	pivot = a[0]
	j = 0
	for index, i in enumerate(a[1:]):
		if i < pivot:
			t = a[j + 1]
			a[j + 1] = pivot
			a[j] = i
			a[index + 1] = t
			j += 1
			print i, j, index, pivot, a
	return quick_sort(a[:j]) + [a[j]] + quick_sort(a[j + 1:])


def quick_sort(a, j = 0):
	if len(a) < 2:
		return a 
	pivot = a[j]
	for index, i in enumerate(a[1:]):
		a[j + 1], a[j], a[index + 1], j = pivot, i, a[j + 1], j + 1 if i < pivot else 1 == 1
	return quick_sort(a[:j], 0) + [a[j]] + quick_sort(a[j + 1:], j + 1)

	
def bubble_sort(a):
	for i_index, i  in enumerate(a):
		for j in range(i_index  + 1, len(a)):
			if a[i_index] > a[j]:
				a[i_index], a[j] = a[j], a[i_index]
	return 

[0, 1, 2, 3, 4, 5, 6]

[6, 3, 8, 2  4, 7, 10]


class Tree:
	def __init__(self):
		self.t = [None] * 127
	def parent(self, i):
		return t[(i-1)/2]
	def left(self, i):
		return self.t[2*i + 1]
	def right(self, i):
		return self.t[2*i + 2]
	def add(self, a, pos = 0):
		if not self.t[pos]:
			self.t[pos] = a
		elif a < self.t[pos]:
			if not self.left(pos):
				self.t[2*pos + 1] = a
			else:
				self.add(a, 2*pos + 1)
		else:
			if not self.right(pos):
				self.t[2*pos + 2] = a
			else:
				self.add(a, 2*pos+ 2)
	def print_tree(self, pos = 0):
		self.print_tree( 2*pos + 1) if self.left(pos) else None
		print self.t[pos] if self.t[pos] else None
		self.print_tree( 2*pos + 2) if self.right(pos) else None

def sort(l, a, start = 0, end = -1):
	end = len(l) -1 if end == -1 else end
	pos = start - ((start - end)/2)
	print start, end, pos
	if l[pos] == a:
		return pos
	elif end - start < 2:
		if a == l[start]:
			return start
		elif a == l[end]:
			return end
		else:
			return -1
	else:
		if a > l[pos]:
			return sort(l, a , pos, end)
		else:
			return sort(l, a , start, pos)


class Node:
	def __init__(self, a):
		self.a = a
		self.next = None
	def set_next(self, n):
		self.next = n

class List:
	def __init__(self):
		self.head = None
	def add(self, a):
		if not self.head:
			self.head = Node(a)
		else:
			n = Node(a)
			n.set_next(self.head)
			self.head = n
	def travese(self):
		cur = self.head
		while (cur):
			print cur.a
			cur = cur.next

class List:
	def __init__(self):
		self.head = None
	def add(self, a):
		if not self.head:
			self.head = Node(a)
		elif a < self.head.a:
			n = Node(a)
			n.next = self.head
			self.head = n
		else:
			self._add(a, prev = self.head, cur = self.head)
	def _add(self, a, prev = None, cur = None):
		if not cur:
			n = Node(a)
			prev.next = n
		elif a < cur.a:
			n = Node(a)
			prev.next = n
			n.next = cur
		else:
			prev = cur
			cur = cur.next
			self._add(a, prev, cur)
	def travese(self):
		cur = self.head
		while (cur):
			print cur.a
			cur = cur.next
			

a = [
[1,2,3],
[4,5,6],
[7,8,9]
]

[
[7,4,1],
[8,5,2],
[9,6,3]
]

def rotate(m):
	for i in range( (len(m) + 1 )/2):
		displacement = len(m) - i*2
		for j in range(i, i + displacement -1):
			m[i + j][displacement -1], m[displacement -1 ][displacement -j - 1], m[displacement -j - 1][i], m[i][j] = m[i][j], m[i + j][displacement -1], m[displacement -1 ][displacement -j - 1], m[displacement -j - 1][i]

1 2 3 4 5
1 2 3 4

2,3

8
8 + 8/4 
12 + 


class Stack:
	def __init__(self):
		self.s = []
	def push(self, a):
		self.s.append(a)
	def pop(self):
		return self.s.pop()

class Queue:
	def __init__(self):
		self.s = []
	def push(self, a):
		self.s.insert(0, a)
	def pop(self):
		return self.s.pop()	


class Queue:
	def __init__(self):
		self.push_stack = Stack()
		self.pull_stack = Stack()
	def add(self, a):
		self.push_stack.push(a)
	def get(self):
		if self.pull_stack.s:
			return self.pull_stack.pop()
		else:
			if self.push_stack.s:
				while self.push_stack.s:
					self.pull_stack.push(self.push_stack.pop())
			if self.pull_stack.s:
				return self.pull_stack.pop()
			else:
				raise Exception('Empty stack')

class Heap:
	def __init__(self):
		self.t = [None] * 127
	def parent(self, i):
		return t[(i-1)/2]
	def left(self, i):
		return self.t[2*i + 1]
	def right(self, i):
		return self.t[2*i + 2]
	def add(self, a):
		if not self.t[0]:
			self.t[0] = a
		else:
			_add(a, 0)
	def right_pos(self, i):
		return 2*i + 2
	def left_pos(self, i):
		return 2*i + 1

def _add(a, pos):
	a > self.t[pos]:
		if a < self.right and a < self.left:
			if self.right(pos):
				t[self.right_pos] = a
				self._add(t, self.right_pos)
		elif a < self.left
		elif a < self.left(pos) and self.left(pos):
			t = self.t[2*i + 1]
			self.t[2*i + 1] = a
			self._add(t, 2*i + 1)
		elif not self.left:
			self.t[2*i + 1] = a
		elif not self.right:
			self.t[2*i + 1] = a
		else:




			if not self.left(pos):
				self.t[2*pos + 1] = a
			else:
				self.add(a, 2*pos + 1)
		else:
			if not self.right(pos):
				self.t[2*pos + 2] = a
			else:
				self.add(a, 2*pos+ 2)
	def print_tree(self, pos = 0):
		self.print_tree( 2*pos + 1) if self.left(pos) else None
		print self.t[pos] if self.t[pos] else None
		self.print_tree( 2*pos + 2) if self.right(pos) else None

def convert_edge_list_to_adjacency_list(g):
	adj_list = {}
	for i in g:
		if i[0] in adj_list:
			adj_list[i[0]] += [(i[1], i[2])]
		else:
			adj_list[i[0]] = [(i[1], i[2])]
	return adj_list

def find_min(g, d):
	cur_min = -1
	min_vertext = None
	for i in d:
		if cur_min == -1 and d[i][0] != -1 and not d[i][2] and i in g:
			cur_min = d[i][0]
			min_vertext = i
		elif d[i][0] != -1 and d[i][0] < cur_min and not d[i][2] and i in g:
			cur_min = d[i][0]
			min_vertext = i
	return min_vertext



def dijkstra(H, a, g):
	G =  convert_edge_list_to_adjacency_list(H)
	shortest_distance = {}
	for i in H:
		shortest_distance[i[0]] = [-1, [a], False]
		shortest_distance[i[1]] = [-1, [a], False]
	current_vertex = a
	shortest_distance[a] = [0, [a], True]
	shortest_distance_from_current_node = 0
	while True:
		for node in G[current_vertex]:
			distance = shortest_distance_from_current_node + node[1]
			print shortest_distance[node[0]][0], distance, 
			if shortest_distance[node[0]][0] == -1 or shortest_distance[node[0]][0]  > distance:
				shortest_distance[node[0]] = [distance, shortest_distance[current_vertex][1]  + [node[0]], False]
		v = find_min(G, shortest_distance)
		shortest_distance[v][2] = True
		shortest_distance_from_current_node = shortest_distance[v][0]
		if not v:
			return []
		if v == g:
			return shortest_distance[g][1]
		else:
			current_vertex = v




def dijkstra(H, a, g):
	G =  convert_edge_list_to_adjacency_list(H)
	shortest_distance = {}
	for i in H:
		shortest_distance[i[0]] = -1 
		shortest_distance[i[1]] = -1 
	current_vertex = a
	shortest_distance[a] = 0
	shortest_distance_from_current_node = 0

	while True:
		for node in G[current_vertex]:
			distance = shortest_distance_from_current_node + node[1]
			print shortest_distance[node[0]][0], distance, 
			if shortest_distance[node[0]][0] == -1 or shortest_distance[node[0]][0]  > distance:
				shortest_distance[node[0]] = [distance, shortest_distance[current_vertex][1]  + [node[0]], False]
		v = find_min(G, shortest_distance)
		shortest_distance[v][2] = True
		shortest_distance_from_current_node = shortest_distance[v][0]
		if not v:
			return []
		if v == g:
			return shortest_distance[g][1]
		else:
			current_vertex = v


class Heap:
    array = []
    # Allows "if item in tree:"
    def __contains__(self, item):
        return self._contains(item, 1)
    def _contains(self, item, index):
        if item > self.array[index-1]:
            return False
        elif item < self.array[index-1]:
            if len(self.array) > index * 2:
                return self._contains(item, index*2) or self._contains(item, index*2+1)
            elif len(self.array) > index * 2 + 1:
                return self._contains(item, index*2)
            else:
                return False
        else:
            return True
    def add(self, item):
        self.array.append(item)
        self._bubble(len(self.array)-1)
    def _bubble(self, index):
        # Note: we take 1-based indexes, but convert them to 0-based ones
        index = index-1
        if self.array[index] > self.array[index/2]:
            tmp = self.array[index]
            self.array[index] = self.array[index/2]
            self.array[index/2] = tmp
            self._bubble(index/2)
    # Useful for debugging. Prints the whole tree
    def __repr__(self):
        return self._repr(1)
    def _repr(self, index, indent=0):
        # Note: we use 1-based indexes here
        string = ""
        string = ('\t' * indent) + str(self.array[index-1]) + '\n'
        if len(self.array) >= index * 2:
            string += ('\t' * (indent+1)) + 'Left:\n'
            string += self._repr(index*2, indent+1)
        if len(self.array) >= index * 2 + 1:
            string += ('\t' * (indent+1)) + 'Right:\n'
            string += self._repr(index*2+1, indent+1)
        return string			
			


class Heap:
	array = []
	def add(self, a):
		self.array.append(a)
		self._bubble(len(self.array) -1)
	def _bubble(self, pos):
		parent = pos/2
		current = pos 
		if current >= 0 and parent >= 0 and self.array[parent] < self.array[current]:
			self.array[current], self.array[parent] = self.array[parent], self.array[current]
			self._bubble(pos/2)
	def get(self):
		m = self.array[0]
		last = self.array.pop()
		self.array[0] = last
		self._bubble_down(0)
		return m
	def _bubble_down(self, pos):
		parent = pos
		left_child = (pos*2) + 1
		right_child = (pos *2) + 2
		print self.array, parent, left_child, right_child
		if parent < len(self.array) and left_child < len(self.array)  and  self.array[parent] < self.array[left_child] and self.array[left_child] > self.array[right_child]:
			self.array[parent],  self.array[left_child] = self.array[left_child], self.array[parent]
			self._bubble_down(pos*2 + 1)
		elif parent < len(self.array) and right_child< len(self.array) and  self.array[parent] < self.array[right_child]:
			self.array[parent],  self.array[right_child] = self.array[right_child], self.array[parent]
			self._bubble_down(pos*2 +  2)


def is_palyndrome(a):
  a = list(str(a))
  for index, i in enumerate(a):
  	if a[index] != a[len(a) - index - 1]:
  		return False
  	if i == len(a) /2:
  		return True
  return True

a = []
for i in reversed(range(999)):
  for j in reversed(range(999)):
    if is_palyndrome(i * j):
      a.append(i*j)

h = []

def largest(a, h):
	for index, i in enumerate(a):
		if index < 100:
			heappush(h, i)
		else:
			if i > h[0]:
				heappop(h)
				heappush(h, i)
	return h


select Name from Employees e
join (select EmployeeId, Salary from Employees) b
where e.BossId = b.EmployeeId
and e.Salary > b.Salary;

select d.Name, max(Salary) from Employees e
join Department d
where e.DepartmentID = d.DepartmentID
group by 1

select d.Name, sum(case when e.Name is not null then 1 else 0) from Employees e
full outer join Department d
where e.DepartmentID = d.DepartmentID
group by 1
having sum(case when e.Name is not null then 1 else 0)  < 3

select d.Name, sum(case when e.Name is not null then 1 else 0) from Employees e
full outer join Department d
where e.DepartmentID = d.DepartmentID
group by 1;



0 -> 1 1
1 -> 2 10
0- -> 2 5


def comb(a):
	letters = { str(i + 1): chr(97 + i)  for i in range(26)}
	paths = set([])
	def combination(a):
		if len(a) == 2 :
			if int(a) < 27:
				yield [a[0], a[1]]
				yield [a]
			else:
				yield [a[0], a[1]]
		elif len(a) == 1:
			yield [a]
		else:
			if int(a[:2]) < 27:
				for i in combination(a[2:]):
					yield [a[0], a[1]] + i
					yield [a[:2]] + i
			else:
				for i in combination(a[2:]):
					yield [a[:2]] + i
			for i in combination(a[1:]):
				yield [a[:1]] + i
	for i in list(combination(a)):
		paths.add( ''.join(map(lambda x : letters[x], i) ))
	return paths

for i in list(combination(a)):
	paths.append( map(i, lambda x : letters[x]) )

def reverseWords(s):
	return s[::-1]

		
def add(l, a):
	zeros = []
	for i in l:
		if i != 0:
			break
		else:
			zeros.append(i)
	return zeros + map(int, list(str(int(''.join( map(str, l) )) + a)))


import itertools
def add(a, b):
	c = sorted(a + b, key = lambda x : x[0])
	f = []
	cur = c[0]
	j = 1
	while(True):
		if j == len(c):
			f.append(cur)
			break
		#for j, index in enumerate(c[i + 1:]):
		if c[j][0] <= cur[1]:
			cur = (cur[0], c[j][1])
			j += 1
			continue
		if c[j][0] > cur[1]:
			f.append(cur) 
			cur = c[j]
			j += 1
			continue
	return f


#minimum consecutive sub string
def min_sub(s, t):
	so = sorted(t)
	min_substring = ''.join(['d'] *len(s))
	for i in range(len(s) - len(t) + 1):
		for j in range( i + len(t), len(s) + 1):
			if ''.join(so) in ''.join(sorted(s[i:j])) and j - i < len(min_substring):
				min_substring = s[i:j]
	return min_substring


min_sub('adobecodebanc', 'abc')



# moving average

def moving_average(l, window):
	a = []
	for i in range(len(l) - window + 1):
		a.append(sum(l[i:i + window])/window)
	return a


extends Configured implements Tool
	static class Mapper externds Mapper<Text, Text, Text, Text> {
		map(Text a, Text b, Context c) {
			c.write(1,2)
		}
			static class Reducer externds Reducer<Text, Text, Text, Text> {
		reduce(Text a, Text b, Context c) {
			c.write(1,2)
		}

		run  {

		}
	}

def consecutive_sum(a, s):
	return _consecutive_sum(a, s, s)

def _consecutive_sum(a, s, s_orig = None, path = None):
	if not path:
		path = []
	if len(a) == 1:
		path.append(a[0])
		return (a[0] == s, path)
	if len(a) == 0:	
		return (False , [])
	print a, s
	if a[0] < s:
		r = _consecutive_sum(a[1:], s - a[0], path = path + [a[0]])
		if r[0]:
			return r
		elif s_orig:
			r = _consecutive_sum(a[1:], s_orig)
			if r[0]:
				return r
			else:
				return (False, [])
		else:
			return (False, [])
	elif a[0] == s:
		return (True, path + [a[0]])
	else:
		return (False, [])
	

from bitarray import bitarray
def get_dupe(a):
	b = bitarray(1024 * 1024 * 1024)
	for i in a:
	  if b[i]:
	    yield i
	  else:
	    b[i] = True


def add(a, b):
	return bin(int(a, 2) + int(b, 2))[2:]


def sort_rotate(s, i):
	a = 0, b = len(s)/2, c = len(s) -1
	_sort(s, a, b, c, i)


def _sort(s, i, a=None, b=None, c=None):	
	a = a or 0
	b = b or len(s)/2
	c = c or len(s) -1
	print s, a, b, c
	if i  == s[b]:
		return b
	elif i == s[a]:
		return a
	elif i == s[c]:
		return c
	if (b -a < 2 and c - b < 2) or a < 0 or b < 0 or c < 0:
		return None
	elif i > s[a] and (s[b] < s[a] or s[b] > i):
		return _sort(s, i, a, a + (b-a)/2, b)
	else:
		return _sort(s, i, b, b + (c-b)/2, c)

def flip(a, b):
	x = list(bin(a)[2:])
	y = list(bin(b)[2:])
	if len(y) > len(x):
		x = ['0' for i in range(len(y) - len(x))] + x
	elif len(x) > len(y):
		y = ['0' for i in range(len(x) - len(y))] + y
	print x,y
	return len([i for i in [ i != y[index] for index, i in enumerate(x)] if i])






def subsets(a):
	for i in range(len(a) + 1):
		if i == 0:
			yield []
		elif i == 1:
			for j in a:
				yield [j]
		else:
			for k in a:
				for j in subsets(a - set([k])):
					yield [k] + j

l = []
def comb(a,n):
	if n == 0:
		yield []
	if n == 1:
		for i in a:
			yield [i]
	else:
		for i in range(len(a)):
			for j in comb(a[i + 1:], n - 1):
				yield [a[i]] + j

def comb(a,n):
	if n == 0:
		yield []
	if n == 1:
		for i in a:
			yield [i]
	else:
		for i in range(len(a)):
			for j in comb(a[i + 1:], n - 1):
				yield [a[i]] + j

def subsets(a):
	for i in range(len(a) + 1):
		for j in comb(list(a), i):
			yield j


def subsets(a):
	def get_item(b, a):
		print b, a
		t = []
		for index, i in enumerate(b):
			if i:
				print index, i, a[index]
				t.append(a[index])
		return t
	for i in range(pow(2, len(a))):
		print  list(bitarray(format(i, '#0%sb' % (len(a) + 2))[2:]))
		yield get_item(list(bitarray(format(i, '#0%sb' % (len(a) + 2))[2:])), a) 

((() ()))

def paranthesis(a):
	if a == 0:
		yield None
	elif a == 1:
		yield '()'
	else:
		for i in paranthesis(a - 1):
			print i
			print len(i)
			for j in range(len(i)):
				yield i[:j] + '()' + i[j:]

d = ['cat', 'cot', 'cog', 'dog']

g = {}

for index,i in enumerate(d):
	print find_all_words_with_one_difference(i, index + 1)
	for j in find_all_words_with_one_difference(i, index + 1):
		if i in g:
			g[i].append(j)
		else:
			g[i] = [j]


def bfs(g, start, end):
	s  = []
	path = []
	s.append(start)
	distance = {}
	distance[start] = 0
	while (s):
		visited_node = s.pop()
		path.append(visited_node) if not visited_node in path else None
		if visited_node == end:
			break
		[s.insert(0, n) for n in g[visited_node] if n not in path] if visited_node in g else None
		for n in g[visited_node]:
			if n not in path:				
				distance[n] = distance[visited_node] + 1 
	return (path, distance)

def find_all_words_with_one_difference(a, index):
	b = []
	for i in d[index:]:
		if len(i) == len(a) and len(set(list(i)) - set(list(a))) == 1:
			b.append(i)
	return b

def find_all_words_with_one_difference(a, steps):
	b = []
	for i in d:
		if len(i) == len(a) and len(set(list(i)) - set(list(a))) == 1 and i not in steps:
			b.append(i)
	return b


def transform(a, b, steps = None):
	print  a, b
	print steps
	if not steps:
		steps = []
	if  a == b:
		return len(steps) 
	else:
		steps.append(a)
		next_words = find_all_words_with_one_difference(a, steps)
		print next_words
		if next_words:
			for i in next_words:
				b = transform(i, b, steps)
				if b:
					return b
				else:
					return None
		else:
			return None


a = ['C', 'D', 'E', 'F', 'G']
b = [3,0,4,1,2]

a[3], a[0] = a[0] , a[3]

[F, D, E, C, G]
[1, 0, 4, 3, 2]


a = ['C', 'D', 'E', 'F', 'G']
b = [3,0,4,1,2]
for i in range(len(b)):
	j = 0
	k = b[0]
	if j == k:
		j = i
		k = b[i]
	a[j], a[k] = a[k], a[j]
	b[j], b[k] = b[k], b[j]

transform('cat', 'dog')


def pairs(l, a):
	i = 0
	j = 1
	while True:
		if l[j] - l[i] == a:
			yield (l[i], l[j])
			i = i + 1
			j = j + 1
		elif l[j] - l[i] > a:
			i = i + 1
			j = i + 1
		else:
			j = j + 1
		if i == j or i > len(l) -1 or j > len(l) - 1:
			break

from itertools import combinations
def extact_sum(N, M, SUM):
	for i in combinations(N, M):
		if sum(i)  == SUM:
			return i
	return []







More customizable wall, not all features used.
More ways to organize
Privacy settings



Interviewer:
Talk about the experience here,
How is the management style, how is the work pressure, work hours
How is it different from the previous challeneges
What is the advice you give, what are things I should be careful
What you like about than other companies you worked
How is the culture different from the other places
Tell me about the team
Performace review process
Do we get training oppurtunities outside? 
DAy to day responsibilities
Expectations for first year

Biggest challeneges

Vision of the company,
Visio of founder,
Company culture
Initiatives
Innovative
Occulus

challeneges
Gap WH first few days

Failures
Single point of failure
GAE

Chalenges when you joined Facebook, adapting to culutre




Single point of failure



_sort([15,16,19,20,25,1,3,4,5,7,10,14], 5)


def min_meeting_rooms(s):
	s = [(i, False) for i in sorted(s, key = s[0])]
	l = len(s)
	n = 0
	i = 0
	j = 0
	current_room = None
	current_schedule = None
	while (True):
		if (j == l - 1):
			break
		if not current_room:
			current_room = 1
			current_schedule = [s[0][i], s[0][j]]
		elif s[i]

(2,5), (5,7), (3,9), (1,12), (4,5) (2,3)

2 -5 5 -7 3 -9 1 -12 4 -5 2 -3
1 2 2 -3 3  4  -5 -5 5  -7  -9 -12

def is_collision(s):
	all_schedueles = { (i*30) : False  for i in range(48)}
	for i in s :
		for j in range(i[0] * 60, int((i[1] - 0.5) * 60), 30):
			if all_schedueles[j]:
				return False
			else:
				all_schedueles[j] = True
	return True


#Ways to climb strair
def stairs(n):
	if n == 0:
		yield ''
	elif n == 1:
		yield '1'
	else:
		for i in stairs(n - 1):
			yield '1' + i
		for i in stairs(n - 2):
			yield '2' + i


#move grid
def move(s, g):
	i, j =  s
	m, n = g
	x = n - j
	y = m - i




def bfs(g, start):
	s = Queue()
	s.put(start)
	distance = {}
	for i in g:
		distance[i] = -1
	distance[start] = 0
	visited = set([])
	while not s.empty():
		visited_node = s.get()
		visited.add(visited_node)
		for n in g[visited_node]:
			if n not in visited:
				s.put(n)		
				distance[n] = distance[visited_node] + 6
	del distance[start]
	return distance.values()


def is_sequence(a, t):
	if a[0] == t:
		return True
	i = 0
	j = 1
	while (True):
		if i == len(a) or j == len(a):
			return False
		s = sum(a[i : j + 1]) 
		if s == t:
			return True
		elif s < t:
			j += 1
		else:
			i += 1

from operator import mul
from functools import reduce
def evaluate(s):
	sum = 0
	for i in s.split('+'):
		sum += reduce(mul, map(int, i.split('*')), 1)
	return sum


def add(l, a):
	new_range_index = None
	index_merged = False
	for index, i in enumerate(l):
		if new_range_index:
			print l[new_range_index]
		if a[0] >= i[0] and a[1] <= i[1]:
			return l
		elif a[0] >= i[0] and a[1] > i[1]:
			l[index] = (i[0], a[1])
			new_range_index = index
			index_merged = True
		elif new_range_index is not None and i[0] >= l[new_range_index][0] and i[1] <= l[new_range_index][1]:
			del l[index]
		elif new_range_index is not None and i[0] >= l[new_range_index][0] and i[1] > l[new_range_index][1]:
			l[new_range_index] = (l[new_range_index][0], i[1])
			del l[index]
	if not index_merged:
		l.append(a)
	return l


def max_difference(a):
	min = a[0]
	max = a[1]
	i = 1 
	j = 2
	while True:
		if i == len(a) or j == len(a):
			break
		if a[i] < min:
			min = a[i]
		i += 1
		if a[j] > max:
			max = a[j]
		j += 1
	return max - min

def inplace_reverse(a):
	words = a.split(' ')
	l = len(words)
	for i in range(len(words)/2):
		words[i], words[l - 1 - i] = words[l - 1 - i], words[i]
	return ' '.join(words)

def inplace_reverse(a):
	words = a.split(' ')
	l = len(words)
	for i in range(len(words)/2):
		words[i], words[l - 1 - i] = words[l - 1 - i], words[i]
	return ' '.join(words)


def sum_2(l, a):
	print l,a
	s = set([])
	for i in l:
		if i in s:
			return True
		elif i < a:
			s.add(a - i)
	return False

def sum_3(l, a):
	for index,i in enumerate(l):
		if i < a:
			if sum_2(l[index + 1:], a - i):
			  return True 
	return False

4
1 2 3 4 5 6



for i in 5 , 4:
(n - 4) [(n-2) * (n -3) /2 ]

2 * 6

3

1 

2 1




1234
1235
1345


2345

12 13 14
23  24
3 4

n-1 , 1

(n- 1)*n/2

4 * 3 * 

[a b c d
 e f g h
 i j k l
 m n o p]

constant

a[]


1,2,3,4,9

1,1
1,2
1,3
1,4
1,5
1,10


ghhgghklgggggg

import sys
from itertools import permutations, combinations

s = sys.stdin.readline().strip()

d = {}
c = set([])
for index, i in enumerate(list(s)):
    if i in d:
        d[i].append(index)
    else:
        d[i] = [index]

def find_d(a,b):
    d = {}
    for index, i in enumerate(list(s[a:b])):
        if i in d:
            d[i] += 1
        else:
            d[i] = 1


def find_d(a,b):
    d = {}
    for index, i in enumerate(list(s[a:b])):
        if i in d:
            d[i] += 1
        else:
            d[i] = 1


def arrange_mid(a, b):
    u = find_d(a + 1,b)
    t = 0
    for i in u:
        if u[i] > 1:
            t += combinations(range(u[i], 2))
    return t

def arrange_end(a, b):
    u_left = find_d(0,a)
    u_right = find_d(b+1, len(s))
    t = 0
    for i in u_left:
        if i in u_right: 
            t += u_left[i] * u_right[i]
    return t

def arrange_4(l):
    return len(list(combinations(range(l), 4)))
        
for i in s:
    if len(d[i]) < 2:
        del d[i]
    else:
        c.add(i)
t = 0

for i in d:
    if len(d[i]) >=4:
        t += arrange_4(len(d[i]))
        
        for j in combinations(d[i], 2):
        	
            if j[1] == i or j[0] == i:
                continue
            if j[1] - j[0] >=2:
                t += arrange_mid(j[0], j[1], c)
            else:
                t += arrange_end(j[0], j[1], c)
    else:
        for j in combinations(d[i], 2):
            if j[1] - j[0] >=2:
                t += arrange_mid(j[0], j[1], c)
            else:
                t += arrange_end(j[0], j[1], c)

    c.discard(i)

print t