
def f(x):
    return (x - 5) ** 2


def df(x):
    return 2 * (x - 5)

learning_rate = 0.3
x = 0
iterations = 14

for i in range(iterations):
    gradient = df(x)
    x = x - learning_rate * gradient
    print(f"Итерация {i + 1}: x = {x:.4f}, f(x) = {f(x):.4f}")

print(f"Минимум функции приблизительно достигается при x = {x:.4f}")

