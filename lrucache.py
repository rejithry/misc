from collections import deque

class Node:
    def __init__(self):
        pass
        #PASS

class LRUCache:
    # @param capacity, an integer
    def __init__(self, capacity):
        self.cache = {}
        self.recency_list = deque()
        self.size = 0
        self.capacity = capacity
    # @return an integer
    def get(self, key):
        if key in self.cache:
            self.recency_list.remove(self.cache[key][1])
            self.recency_list.appendleft(self.cache[key][1])
            return self.cache[key][0]
        else:
            return -1
    # @param key, an integer
    # @param value, an integer
    # @return nothing
    def set(self, key, value):
        if self.size == self.capacity:
            self.recency_list.pop()
            self.size  -= 1
        n = Node()
        self.cache[key] = (value, n)
        self.recency_list.appendleft(n)
        self.size += 1
        
        

