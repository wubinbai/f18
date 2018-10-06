#!/usr/bin/python
# -*- coding: utf-8 -*-
# Breadth First Search

v = 7  # number of vertices
adj = []
for i in xrange(0, v + 1):
  adj.append([])
# Marking all the vertices as unvisited
visited = [False] * (v + 1)  # checks which nodes are visited


def addEdge(a, b):
  # Creating an edge between a and b
  adj[a].append(b)
  # Creating an edge between b and a
  adj[b].append(a)


def getBFS():
  for i in range(1, v + 1):
    # Call bfs for the unvisited nodes
    if not visited[i]:
      bfs(i)


def bfs(start):
  queue = []
  queue.append(start)
  # mark the start node as visited
  visited[start] = True
  while queue:
    # Dequeue the oldest vertex from queue
    start = queue.pop(0)
    # Printing the dequeued vertex
    print start,
    # Get all the adjacent vertices of the dequeued vertex
    for i in range(len(adj[start])):
      ele = adj[start][i]
      # Enqueue the vertex if unvisited and mark as visited
      if not visited[ele]:
        visited[ele] = True
        queue.append(ele)


for i in range(0, 8):
  adj[i] = []
addEdge(1, 2)
addEdge(1, 3)
addEdge(1, 5)
addEdge(4, 2)
addEdge(6, 4)
addEdge(6, 5)
addEdge(2, 5)

getBFS()
