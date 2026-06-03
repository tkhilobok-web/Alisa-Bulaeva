from collections import deque
import time


class Cell:
    """Элемент связанного списка"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.next = None

# А) Реализация через массив
def count_components_array(grid):
    n = len(grid)
    m = len(grid[0])

    visited = [[False] * m for _ in range(n)]
    components = 0

    for i in range(n):
        for j in range(m):
            if grid[i][j] and not visited[i][j]:
                components += 1
                stack = [(i, j)]

                while stack:
                    x, y = stack.pop()

                    if visited[x][y]:
                        continue

                    visited[x][y] = True

                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = (y + dy) % m  # цилиндр

                        if 0 <= nx < n:
                            if grid[nx][ny] and not visited[nx][ny]:
                                stack.append((nx, ny))

    return components

# Б) Реализация через связанный список
def build_linked_list(grid):
    head = None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j]:
                node = Cell(i, j)
                node.next = head
                head = node
    return head


def count_components_linked(grid):
    n = len(grid)
    m = len(grid[0])

    head = build_linked_list(grid)
    visited = [[False] * m for _ in range(n)]
    components = 0

    cur = head
    while cur:
        if not visited[cur.x][cur.y]:
            components += 1
            stack = [(cur.x, cur.y)]

            while stack:
                x, y = stack.pop()

                if visited[x][y]:
                    continue

                visited[x][y] = True

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = (y + dy) % m

                    if 0 <= nx < n:
                        if grid[nx][ny] and not visited[nx][ny]:
                            stack.append((nx, ny))

        cur = cur.next

    return components

# В) Стандартная библиотека Python
def count_components_std(grid):
    n = len(grid)
    m = len(grid[0])

    alive = {
        (i, j)
        for i in range(n)
        for j in range(m)
        if grid[i][j]
    }

    components = 0

    while alive:
        start = alive.pop()
        components += 1
        q = deque([start])

        while q:
            x, y = q.popleft()

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = (y + dy) % m

                if 0 <= nx < n:
                    neighbour = (nx, ny)
                    if neighbour in alive:
                        alive.remove(neighbour)
                        q.append(neighbour)

    return components


# Функция для создания сетки с удаленными клетками

def create_grid(m, n, removed):
    """Создает сетку с заданными удаленными клетками"""
    grid = [[True] * n for _ in range(m)]
    for x, y in removed:
        # Проверяем, что координаты в пределах сетки
        if 1 <= x <= m and 1 <= y <= n:
            grid[x - 1][y - 1] = False
    return grid

# Бенчмаркинг
def benchmark(func, grid):
    start = time.perf_counter()
    result = func(grid)
    elapsed = time.perf_counter() - start
    return result, elapsed


# Основная функция
def main():
    print("Булаева Алиса Ростиславовна РПИа-о25")
    print("=" * 60)

    m, n = map(int, input("Введите M и N: ").split())
    k = int(input("Введите количество удаленных клеток: "))

    removed = []
    if k > 0:
        print("Введите координаты удаленных клеток (x y):")
        for _ in range(k):
            x, y = map(int, input().split())
            removed.append((x, y))

    # Создаем сетку
    grid = create_grid(m, n, removed)

    # Выводим информацию о сетке
    print("\n" + "=" * 60)
    print(f"Размер сетки: {m} x {n}")
    print(f"Оставшихся клеток: {sum(sum(row) for row in grid)}")
    print(f"Удаленных клеток: {k}")
    print("=" * 60 + "\n")

    # Тестируем три метода
    funcs = [
        ("Массив", count_components_array),
        ("Связанный список", count_components_linked),
        ("Стандартная библиотека", count_components_std)
    ]

    results = []
    for name, func in funcs:
        result, t = benchmark(func, grid)
        results.append((name, result, t))
        print(f"{name:25} компонент = {result:3d} время = {t:.6f} сек")

    # Проверяем согласованность результатов
    print("\n" + "=" * 60)
    if len(set(r[1] for r in results)) == 1:
        print("Все три метода дали одинаковый результат")
    else:
        print("Результаты методов различаются!")
        for name, res, _ in results:
            print(f"  {name}: {res} компонент")


if __name__ == "__main__":
    main()