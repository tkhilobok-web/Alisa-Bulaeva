import time
from collections import deque


#  1. Реализация через массив (стек / DFS) 

def count_components_array(grid, M, N, cylinder=False):
    visited = [[False] * N for _ in range(M)]
    components = 0

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i in range(M):
        for j in range(N):
            if grid[i][j] == 1 and not visited[i][j]:
                components += 1
                stack = [(i, j)]

                while stack:
                    x, y = stack.pop()

                    if visited[x][y]:
                        continue

                    visited[x][y] = True

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy

                        if cylinder:
                            ny = ny % N

                        if 0 <= nx < M and 0 <= ny < N:
                            if grid[nx][ny] == 1 and not visited[nx][ny]:
                                stack.append((nx, ny))

    return components


#  2. Связанный список (очередь / BFS) 
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class QueueLinkedList:
    def __init__(self):
        self.front = None
        self.rear = None

    def enqueue(self, item):
        new_node = Node(item)
        if self.rear is None:
            self.front = self.rear = new_node
            return
        self.rear.next = new_node
        self.rear = new_node

    def dequeue(self):
        if self.front is None:
            return None
        temp = self.front
        self.front = self.front.next

        if self.front is None:
            self.rear = None

        return temp.data

    def is_empty(self):
        return self.front is None


def count_components_linked(grid, M, N, cylinder=False):
    visited = [[False] * N for _ in range(M)]
    components = 0

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i in range(M):
        for j in range(N):
            if grid[i][j] == 1 and not visited[i][j]:
                components += 1
                queue = QueueLinkedList()
                queue.enqueue((i, j))

                while not queue.is_empty():
                    x, y = queue.dequeue()

                    if visited[x][y]:
                        continue

                    visited[x][y] = True

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy

                        if cylinder:
                            ny = ny % N

                        if 0 <= nx < M and 0 <= ny < N:
                            if grid[nx][ny] == 1 and not visited[nx][ny]:
                                queue.enqueue((nx, ny))

    return components


# 3. STL (deque) 
def count_components_stl(grid, M, N, cylinder=False):
    visited = [[False] * N for _ in range(M)]
    components = 0

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for i in range(M):
        for j in range(N):
            if grid[i][j] == 1 and not visited[i][j]:
                components += 1
                queue = deque()
                queue.append((i, j))

                while queue:
                    x, y = queue.popleft()

                    if visited[x][y]:
                        continue

                    visited[x][y] = True

                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy

                        if cylinder:
                            ny = ny % N

                        if 0 <= nx < M and 0 <= ny < N:
                            if grid[nx][ny] == 1 and not visited[nx][ny]:
                                queue.append((nx, ny))

    return components


def main():
    print("====================================")
    print("Автор: Булаева Алиса Ростиславовна")
    print("Группа: РПИА-025")
    print("====================================\n")

    # Пример: шахматный узор (как в задании)
    M, N = 8, 8
    grid = [[(i + j) % 2 for j in range(N)] for i in range(M)]

    print("Размер поля:", M, "x", N)
    print("1 - клетка есть, 0 - удалена\n")

    # --- Массив ---
    start = time.time()
    res1 = count_components_array(grid, M, N)
    t1 = time.time() - start

    # --- Связанный список ---
    start = time.time()
    res2 = count_components_linked(grid, M, N)
    t2 = time.time() - start

    # --- STL ---
    start = time.time()
    res3 = count_components_stl(grid, M, N)
    t3 = time.time() - start

    print("ОБЫЧНЫЙ ЛИСТ:")
    print("Массив (стек):", res1, "время:", round(t1, 6))
    print("Связанный список:", res2, "время:", round(t2, 6))
    print("STL (deque):", res3, "время:", round(t3, 6))

    print("\nЦИЛИНДР:")
    print("Массив:", count_components_array(grid, M, N, True))
    print("Связанный список:", count_components_linked(grid, M, N, True))
    print("STL:", count_components_stl(grid, M, N, True))


if __name__ == "__main__":
    main()
