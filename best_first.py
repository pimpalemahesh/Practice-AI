from queue import PriorityQueue
graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : [],
  '8' : []
}
v = 14

def best_first_search(actual_Src, target, n):
	visited = [False] * n
	pq = PriorityQueue()
	pq.put((0, actual_Src))
	visited[actual_Src] = True
	
	while pq.empty() == False:
		u = pq.get()[1]
		print(u, end=" ")
		if u == target:
			break

		for v, c in graph[u]:
			if visited[v] == False:
				visited[v] = True
				pq.put((c, v))
	print()

def addedge(x, y, cost):
	graph[x].append((y, cost))
	graph[y].append((x, cost))

source = 5
target = 8
best_first_search(source, target, v)
