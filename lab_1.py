import random

class Monkey:
    def __init__(self):
        self.state = "Сидит"
        self.reward_count = 0

    def handshake(self):
        """Обезьяна выполняет команду 'Пожать руку'."""
        self.state = "Пожимает руку"
        return True

    def ignore_command(self):
        """Обезьяна игнорирует команду."""
        self.state = "Сидит"
        return False

    def get_reward(self):
        self.reward_count += 1
        print("Получено лакомство! Всего наград:", self.reward_count)

    def get_negative_feedback(self):
        """Получение отрицательной обратной связи."""
        print( " Неправильно. Попробуй ещё раз.")

monkey = Monkey()

for _ in range(10):
    action = random.choice(["handshake", "ignore"])
    print(f"Обезьяна пытается: {action}")

    if action == "handshake":
        if monkey.handshake():
            monkey.get_reward()
        else:
            monkey.get_negative_feedback()
    else:
        if monkey.ignore_command():
            monkey.get_negative_feedback()
        else:
            print("Обезьяна сидит.")

print("Обучение завершено.")