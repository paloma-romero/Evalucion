#generador de variables random
import random

# Función evaluar calcula el valor total de los objetos seleccionados
#  y compara si el peso total de estos no supera el peso total máximo
def evaluate(items, knapsack_weight, solution):
    total_value = 0 #inicializamo el valor
    total_weight = 0 #inicializamos peso
    for i in range(len(items)): #seleccion de los objetos
        if solution[i] == 1:
            total_value += items[i][0] 
            total_weight += items[i][1]
    if total_weight > knapsack_weight: #comparativa del peso total contra el peso de la mochila
        return [0, total_weight] 
    else:
        return (total_value, total_weight)
        

# Función vecino: las fuhnciones que se vayan obteniendo la guardaremos en la solución vecina a la solucion actual,
def generate_neighbor(solution):
    neighbor = solution[:]
    i = random.randint(0, len(solution)-1)
    neighbor[i] = 1 - neighbor[i]
    return neighbor

# Inicializar la solución: las soluciones las guardara de forma al azar
def initialize_solution(n):
    return [random.randint(0,1) for i in range(n)]

#Algoritmo de Hill climbing: genera soluciones vecinas, evalua las puntuaciones
#  y evalua la mejor solucion entre la solucion actual y solucion vecina.
def hill_climbing(items, knapsack_weight, n_iter):
    current_solution = initialize_solution(len(items))
    current_value, total_weight = evaluate(items, knapsack_weight, current_solution) #obtener la knapsack_weight con cada iteración y la mejor solución presente
    print("El valor actual es: " + str(current_value) +" con un peso de: "+str(total_weight))
    for i in range (n_iter): 
        neighbor = generate_neighbor(current_solution)
        neighbor_value, total_weight = evaluate(items, knapsack_weight, neighbor)
        if neighbor_value > current_value:
            current_value = neighbor_value
            current_solution = neighbor
            print("En la iteracion: " + str(i) + " se escogio: "+ str(current_solution) +" con el valor de: "+ str(current_value)+" con un peso de: "+str(total_weight))
    return current_solution, current_value


# uso con items definidos y peso definido
items = [[4,12],[2,1],[2,2],[1,1],[10,4]]
knapsack_weight = 15
solution, value = hill_climbing(items, knapsack_weight,10)
print("Items seleccionados:", [items[i] for i in range(len(items)) if solution[i] == 1])
print("Valor total:", value) #solucion peso y valor
