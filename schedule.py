"""Partition selected tests using their measured durations."""

from heapq import heapreplace


WORKER_COST_MS = 2_000
MAX_WORKERS = 8


def _lpt_partition(positive_tasks, count):
    workers = [[] for _ in range(count)]
    loads = [(0, worker) for worker in range(count)]
    worker_loads = [0] * count
    for duration, number in positive_tasks:
        load, worker = loads[0]
        workers[worker].append(number)
        new_load = load + duration
        worker_loads[worker] = new_load
        heapreplace(loads, (new_load, worker))
    return workers, worker_loads


def _refine_partition(workers, worker_loads, durations_ms):
    """Improve makespan via greedy move/swap from the heaviest worker."""
    count = len(workers)
    if count <= 1:
        return
    for _ in range(32):
        max_w = max(range(count), key=worker_loads.__getitem__)
        max_load = worker_loads[max_w]
        improved = False

        # Try moving one task from max_w to a lighter worker.
        best_move = None
        best_new_max = max_load
        for idx, t in enumerate(workers[max_w]):
            d = durations_ms[t]
            rem = max_load - d
            for w in range(count):
                if w == max_w:
                    continue
                cand = max(rem, worker_loads[w] + d)
                if cand < best_new_max:
                    best_new_max = cand
                    best_move = (idx, w, d)
        if best_move is not None:
            idx, w, d = best_move
            t = workers[max_w].pop(idx)
            workers[w].append(t)
            worker_loads[max_w] -= d
            worker_loads[w] += d
            improved = True
            continue

        # Try swapping one task from max_w with a smaller task from another worker.
        best_swap = None
        best_new_max = max_load
        for i1, t1 in enumerate(workers[max_w]):
            d1 = durations_ms[t1]
            for w in range(count):
                if w == max_w:
                    continue
                lw = worker_loads[w]
                for i2, t2 in enumerate(workers[w]):
                    d2 = durations_ms[t2]
                    if d2 >= d1:
                        continue
                    diff = d1 - d2
                    cand = max(max_load - diff, lw + diff)
                    if cand < best_new_max:
                        best_new_max = cand
                        best_swap = (i1, w, i2, diff)
        if best_swap is not None:
            i1, w, i2, diff = best_swap
            workers[max_w][i1], workers[w][i2] = workers[w][i2], workers[max_w][i1]
            worker_loads[max_w] -= diff
            worker_loads[w] += diff
            improved = True
            continue

        # Try swapping one heavy task from max_w with two smaller tasks from another worker.
        if len(workers[max_w]) >= 1:
            best_2for1 = None
            best_new_max = max_load
            for i1, t1 in enumerate(workers[max_w]):
                d1 = durations_ms[t1]
                for w in range(count):
                    if w == max_w or len(workers[w]) < 2:
                        continue
                    lw = worker_loads[w]
                    ww = workers[w]
                    # Cap inner scan on huge workers to keep scheduler sub-millisecond
                    lim = min(len(ww), 24)
                    for a in range(lim):
                        da = durations_ms[ww[a]]
                        for b in range(a + 1, lim):
                            db = durations_ms[ww[b]]
                            diff = d1 - (da + db)
                            if diff <= 0:
                                continue
                            cand = max(max_load - diff, lw + diff)
                            if cand < best_new_max:
                                best_new_max = cand
                                best_2for1 = (i1, w, a, b, diff)
            if best_2for1 is not None:
                i1, w, a, b, diff = best_2for1
                t1 = workers[max_w].pop(i1)
                tb = workers[w].pop(b)
                ta = workers[w].pop(a)
                workers[max_w].extend((ta, tb))
                workers[w].append(t1)
                worker_loads[max_w] -= diff
                worker_loads[w] += diff
                improved = True

        if not improved:
            break


def schedule(numbers, durations_ms):
    """Choose the lowest-cost partition for the selected tests across up to 8 workers."""
    tasks = sorted(((durations_ms[number], number) for number in numbers), reverse=True)
    if not tasks:
        return []

    positive_tasks = [task for task in tasks if task[0] > 0]
    if not positive_tasks:
        return [[number for _, number in tasks]]

    total = sum(duration for duration, _ in positive_tasks)
    best_workers = [[number for _, number in positive_tasks]]
    best_cost = total + WORKER_COST_MS

    for count in range(2, min(MAX_WORKERS, len(positive_tasks)) + 1):
        overhead = count * WORKER_COST_MS
        lower_bound = max(positive_tasks[0][0], (total + count - 1) // count) + overhead
        if lower_bound >= best_cost:
            continue

        workers, worker_loads = _lpt_partition(positive_tasks, count)
        if max(worker_loads) + overhead > lower_bound:
            _refine_partition(workers, worker_loads, durations_ms)

        cost = max(worker_loads) + overhead
        if cost < best_cost:
            best_workers = workers
            best_cost = cost

    best_workers[0].extend(number for _, number in tasks[len(positive_tasks):])
    return best_workers


def schedule_cost_ms(numbers, durations_ms):
    """Return the exact simulated cost in milliseconds for scheduling `numbers`."""
    if not numbers:
        return 0
    workers = schedule(numbers, durations_ms)
    if not workers:
        return 0
    largest = max(sum(durations_ms[n] for n in w) for w in workers)
    return largest + WORKER_COST_MS * len(workers)
