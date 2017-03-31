class Node:
  def __init__(self, a):
    self.a = a
    self.l = None
    self.r = None
    self.p = None

class Tree:
  def __init__(self):
    self.root = None
  def insert(self, a):
    if not self.root:
      self.root = Node(a)
    else:
      r = self.root
      self._insert(a, r)
  def _insert(self, a, root):
    n = Node(a)
    if not root.l and a < root.a:
      root.l = n
      n.p = root
    elif not root.r and a >= root.a:
      root.r = n
      n.p = root
    elif a > root.a:
      self._insert(a, root.r)
    else:
      self._insert(a, root.l)
  def traverse(self, root):
    if not root:
      return
    else:
      self.traverse(root.l)
      print root.a
      self.traverse(root.r)
  def successor(self, n):
    if n.r:
      return self.minimum(n.r)
    else:
      return n.p
  def minimum(self, n):
    if not n.l:
      return n
    else:
      return self.minimum(n.l)
  def maximum(self, n):
    if not n.r:
      cur_p = n.p
      while cur_p and cur_p.r == n:
        n = cur_p
        cur_p = n.p
      return cur_p
    else:
      return self.maximum(n.r)
  def predecessor(self, n):
    if n.l:
      return self.maximum(n.l)
    else:
      cur_p = n.p
      while cur_p and cur_p.l == n:
        n = cur_p
        cur_p = n.p
      return cur_p
  def delete(self, a):
    # Simply delete for leaf
    if not a.l and a.r:
      if a == a.p.l:
        a.p.l == None
      else:
        a.p.r = None
    elif not a.l:
      self.transplant(a, a.r)
    elif not a.r:
      self.transplant(a.l)
    else:
      s = self.successor(a)
      if s != a.r:
        self.transplant(s, s.r)
        s.r = a.r
        s.r.p = s
      self.transplant(a, s)
      s.l = a.l
      s.l.p = s
  def transplant(self, u, v):
    if not u.p:
      self.root = v
    elif u == u.p.l:
      print 'here'
      u.p.l = v
      v.p =  u.p
    else:
      u.p.r = v
    if v:
      v.p = u.p
  def find(self, a):
    if not self.root:
      return False
    else:
      def _find(a, root):
        if not root:
          return False
        if a == root.a:
          return True
        elif a > root.a:
          return _find(a, root.r)
        else:
          return _find(a, root.l)
      return _find(a, self.root)
  def find_path(self, a):
    if not self.root:
      return []
    else:
      def _find_path(a, root, current_path):
        if not root:
          return current_path
        if a == root.a:
          return current_path + [root]
        elif a > root.a:
          return _find_path(a, root.r, current_path + [root])
        else:
          return _find_path(a, root.l, current_path + [root])
      return _find_path(a, self.root, [])
  def find_lca(self, n1, n2):
    # Always n1  > n1
    l1 = self.find_path(n1)
    for i in l1:
      if i.a == n1:
        return i.a
      elif i.a > n1 and i.a < n2:
        return i.a
  def height(self, root):
    if not root:
      return 0
    else:
      return 1 + max(self.height(root.r) , self.height(root.l))
  def diameter(self, root):
    if not root:
      return 0
    else:
      return max(max( 1 + self.diameter(root.r), 1 + self.diameter(root.l)), sum([self.height(root.r)  , self.height(root.l)] ) + 1 )



t = Tree()
t.insert(5)
t.insert(3)
t.insert(7)
t.insert(4)
t.insert(2)
t.insert(6)
t.insert(8)
t.find(9)
[i.a for i in t.find_path(2)]
t.find_lca(4,2)
t.height(t.root)
t.diameter(t.root)

t = Tree()
t.insert(1)
t.insert(2)
t.insert(3)
t.insert(4)
t.insert(5)
t.height(t.root)
t.diameter(t.root)


t.traverse(t.root)
t.minimum(t.root)

t.successor(t.root.r.l)
t.predecessor(t.root).a
t.predecessor(t.root.r.l).a

t.delete(t.root.l)


class Node:
  def __init__(self, a):
    self.a = a
    self.next = None
    self.prev = None


class DLL:
  def __init__(self):
    self.head = None
    self.tail = None
    self.count = 0
  def add(self, a):
    n = Node(a)
    if not self.head:
      self.head = n
      self.tail = n
    else:
      self.tail.next = n
      n.prev = self.tail
      self.tail = n
    self.increment()
  def print_list(self):
    cur = self.head
    while cur:
      print cur.a
      cur = cur.next
  def increment(self):
    self.count += 1
  def get_element(self, n):
    cur = self.head
    count = 1
    while count < n:
      cur = cur.next
      count += 1
    return cur
  
  


class TreeNode:
  def __init__(self, a):
    self.a = a
    self.right = None
    self.left = None


class Tree:
  def __init__(self, root):
    self.root = root
  def traverse(self):
    self._traverser(self.root)
  def _traverser(self, root):
    if root:
      if root.left:
        self._traverser(root.left)
      print root.a
      if root.right:
        self._traverser(root.right)


def create_tree(d, start, end):
  if end - start <= 2:
    if end == start:
      return  TreeNode(d.get_element(start).a) 
    elif end - start == 1:
      t = TreeNode(d.get_element(start).a)
      t.right = TreeNode(d.get_element(end).a)
      return t
    elif end - start == 2:
      t = TreeNode(d.get_element(start + 1).a)
      t.left = TreeNode(d.get_element(start).a)
      t.right = TreeNode(d.get_element(start + 2).a)
      return t
  else:
    mid = start + (end - start) /2
    t = TreeNode(d.get_element(mid).a)
    t.left = create_tree(d, start, start + ((end - start)/2) - 1 )
    t.right = create_tree(d, end - ((end - start)/2) + 1 , end)
    return t


def create_dll(root):
  d = DLL()
  _create_dll(root, d)
  return d


def _create_dll(root, d):
  if root.left:
    _create_dll(root.left, d)
  d.add(root.a)
  if root.right:
    _create_dll(root.right, d)



def middle_element(d):
  cur = d.head
  cur_2 = d.head
  cur_2_pass = False
  while cur:
    cur = cur.next
    if cur_2_pass:
      cur_2 = cur_2.next
      cur_2_pass = False
    else:
      cur_2_pass = True
  print cur_2.a


def duplicate_sub_tree(root):
  if not root:
    return Dalse
  is_dupe = False
  if root.left:
    is_dupe = is_dupe || duplicate_sub_tree(root.left)
  if root.rigth:
    is_dupe = is_dupe || duplicate_sub_tree(root.rigth)
  return is_dupe


def product(d):
  


d = DLL()
d.add(1)
d.add(2)

d.add(3)
d.add(4)
d.add(5)
d.add(6)
d.add(7)
d.add(8)
d.add(9)
d.add(10)
d.add(11)
d.print_list()


t = Tree(create_tree(d, 1, d.count))
