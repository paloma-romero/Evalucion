import random

# Función para calcular el valor de una solución
def calc_value(solution, values):
    return sum([values[i] for i in range(len(solution)) if solution[i]])

# Función para calcular el peso de una solución
def calc_weight(solution, weights):
    return sum([weights[i] for i in range(len(solution)) if solution[i]])

def grasp(num_items, max_weight, values, weights, alpha):
    # Inicializar solución vacía
    solution = [0] * num_items
    best_solution = solution.copy()
    best_value = 0

    # Iterar hasta que se alcance el criterio de parada
    while True:
        # Construir solución candidata
        candidate_set = [i for i in range(num_items) if solution[i] == 0]
        for i in candidate_set:
            if random.random() < alpha:
                candidate_set.append(i)
        if not candidate_set:
            break
        item = max(candidate_set, key=lambda x: values[x] / weights[x])

        # Añadir el elemento seleccionado a la solución
        solution[item] = 1

        # Actualizar la mejor solución
        if calc_weight(solution, weights) <= max_weight:
            solution_value = calc_value(solution, values)
            if solution_value > best_value:
                best_value = solution_value
                best_solution = solution.copy()
        # Revisar si cumple criterio de parada
        if best_value >= stop_criterion:
            break
    return best_solution, best_value
