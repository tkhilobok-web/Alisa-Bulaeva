def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

# Информация о студенте
print("ФИО: Булаева Алиса Ростиславовна")
print("Группа: РПИа-о25")

try:
    a, b, c = map(int, input().split())
except:
    print("Impossible")
    exit()

g, x0, y0 = egcd(a, b)

if c % g != 0:
    print("Impossible")
    exit()

x0 *= c // g
y0 *= c // g

step_x = b // g
step_y = -a // g

if step_x > 0:

    if x0 < 0:
        k = (-x0 + step_x - 1) // step_x
    else:
        k = -(x0 // step_x)

    best_x = None
    best_y = None

    for k_try in [k, k + 1]:
        x = x0 + k_try * step_x
        y = y0 + k_try * step_y
        if x >= 0 and (best_x is None or x < best_x):
            best_x = x
            best_y = y

    print(best_x, best_y)
else:
    if x0 >= 0:
        print(x0, y0)
    else:
        print("Impossible")