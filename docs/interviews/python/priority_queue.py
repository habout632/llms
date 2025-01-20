class PriorityQueue:

    def __init__(self):
        self.queue = []

    def insert(self, item, priority):
        self.queue.append((priority, item))
        self.queue.sort(key=lambda x: x[1], reverse=True)

    def remove(self):
        if not self.queue:
            raise Exception("Queue is empty")
        return self.queue.pop[0][0]

    def peek(self):
        if not self.queue:
            raise Exception("Queue is empty")
        return self.queue[0][0]

    def is_empty(self):
        return len(self.queue) == 0


if __name__ == '__main__':
    pq = PriorityQueue()
    pq.insert(1, 3)
    pq.insert(1, 1)
    pq.insert(1, 2)