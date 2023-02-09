# Copyright Esteban Mendiola Tellez All Rights Reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file at root of this proyect or in
# https://gitlab.com/TebanMT/hill-climbing/-/blob/master/LICENSE
# ==============================================================================
from random import seed
from random import randint
from numpy import asarray
from numpy.random import randint
import functools
import copy
import time

from util import write_specific_line, txt_decorator

seed(1)

class Item(object):
    """ La clase Item construye un objeto item para ser utilizado en la clase Mochila
    preferentemente.

    El constructor toma los sguientes parametros:

    id         : int id del item
    valor_item : int Valor del item
    peso_item  : int Peso del item
    """
    def __init__(self, id, valor_item, peso_item):
        self.id = id
        self.valor_item = valor_item
        self.peso_item = peso_item

    def __eq__(self, other):
        if self.id == other.id and self.valor_item == other.valor_item and self.peso_item == other.peso_item :
            return True
        else:
            return False

    def __str__(self):
        return "Item: {}, Valor: {}, Peso: {}".format(self.id,self.valor_item, self.peso_item)



class Mochila(object):
    """La clase Mochila implementa el algoritmo hill-climbing random para solucionar
    el problema llamado 'problema de la mochila'

    El constructor toma los sguientes parametros:

    max_peso : int Peso maximo que soporta la mochila
    items    : Items class Lista de items que se construyen a partir de la clase Item
    max_iter : int Numero maximo de iteraciones para el algoritmo
    """
    def __init__(self, max_peso, items, max_iter):
        self.max_peso = max_peso
        self.items = items
        self.max_iter = max_iter

    @txt_decorator
    def solve_rmhc(self, *args, **kwargs):
        """solve(False/True) esta funcion ejecuta el algoritmo hill-climbing random

        Parameters
        ----------
        *args[0] : booleano
            True si se desea obtener la lista de objetos que dan soloucion al problema, de lo contrario
            False.

        Returns
        -------
        dict
            Un diccionario con las siguientes clave-valor:
                solucion_objs   : lista de objectos Item que maximizan el valor de la mochila (condicionado a args[0])
                solucion_items  : lista de diccionarios con las propiedades de los items que maximizan el valor de la mochila
                solucion_binary : lista de 1/0 en las posiciones de los items que maximizan el valor de la mochila
                                  0 significa que el item no se encuentra en la solucion, 1 si se encuentra
                peso            : int es el peso que tienen todos los items que dan solucion al problema
                valor           : int es el valor que tienen todos los items que dan solucion al problema

        Raises
        ------
        IndexError : si el argumento args[0] no se proporciona
        """
        f = kwargs['file']
        solucion = self.initial_solution(only_binary=False)
        solucion_eval = self.evaluar(solucion['obj'][:])
        f.write("\nSolucion Inicial: "+str(solucion['binary'])+"\nPeso Inicial: "+str(solucion_eval['peso'])+'\nValor Incial: '+str(solucion_eval['valor'])+'\n\n===========ITERACIONES==============\n\n')
        # run the hill climb
        for i in range(self.max_iter):
            candidato = self.random_neighbor(copy.deepcopy(solucion))
            candidato_eval = self.evaluar(candidato['obj'][:])
            message = str(i+1)+'.- Candidato: '+str(candidato['binary'])+ ' con Peso: '+str(candidato_eval['peso'])+ ' y Valor: '+str(candidato_eval['valor'])+ ' > a solucion actual: '+str(solucion['binary'])+ ' con Peso: '+str(solucion_eval['peso'])+ ' y Valor: '+str(solucion_eval['valor'])
            if candidato_eval["valor"] > solucion_eval["valor"]:
                solucion, solucion_eval = candidato, candidato_eval
                message += ' --> El candidato es mejor solucion'
            message += '\n'
            f.write(message)
        #end the hill climb
        try:
            return {
                "solucion_objs"   : solucion['obj'] if args[0] is not False else '',
                "solucion_items"  : [{"Item":x.id,"Valor":x.valor_item,"Peso":x.peso_item} for x in solucion['obj']],
                "solucion_binary" : solucion['binary'],
                "peso"            : solucion_eval['peso'],
                "valor"           : solucion_eval['valor']
            }
        except IndexError:
            raise Exception("Necesita especificar si requiere el arreglo de objetos Items (True o False en el primer parametro)")

    def solve_sa(self, *args, **kwargs):
        """"""
        import numpy as np
        solucion = self.initial_solution(only_binary=False)
        solucion_eval = self.evaluar(solucion['obj'][:])
        temperatura = solucion_eval["valor"] * 0.4
        while temperatura >= 0.1:
            for _ in range(self.max_iter):
                candidato = self.random_neighbor(copy.deepcopy(solucion))
                candidato_eval = self.evaluar(candidato['obj'][:])
                delta =  candidato_eval["valor"] - solucion_eval["valor"]
                if np.random.uniform(low=0.0, high=1.0, size=None) <= np.power(np.e,(-delta/temperatura)) or delta > 0:
                    solucion, solucion_eval = candidato, candidato_eval
            temperatura = temperatura * np.random.uniform(low=0.8, high=0.99, size=None)
        return {
                "solucion_objs"   : solucion['obj'] if args[0] is not False else '',
                "solucion_items"  : [{"Item":x.id,"Valor":x.valor_item,"Peso":x.peso_item} for x in solucion['obj']],
                "solucion_binary" : solucion['binary'],
                "peso"            : solucion_eval['peso'],
                "valor"           : solucion_eval['valor']
            }

    def solve_ga(self, num_individuos, num_generaciones=100):
        """"""
        import algos_geneticos as g
        def evaluar_ga(*args):
            sum_valor = 0
            sum_peso = 0
            for i,x in enumerate(args):
                if x == 1:
                    sum_valor += self.items[i].valor_item
                    sum_peso += self.items[i].peso_item
            if sum_peso > self.max_peso:
                return -float("inf")
            return sum_valor
        poblacion = g.Poblacion(num_individuos, len(self.items),limites_inf=0, limites_sup=1, repre=self.initial_solution, verbose=True)
        poblacion.optimizar(evaluar_ga, num_generaciones=num_generaciones)
        solucion_obj = [self.items[i] for i,v in enumerate(poblacion.cromosoma_optimo) if v==1]
        return {
                "solucion_items"  : [{"Item":x.id,"Valor":x.valor_item,"Peso":x.peso_item} for x in solucion_obj],
                "solucion_binary" : poblacion.cromosoma_optimo,
                "peso"            : self.evaluar(solucion_obj)["peso"],
                "valor"           : poblacion.valor_funcion_optimo,
                "Valor_ge"        : self.evaluar(solucion_obj)["valor"]
            }

    def evaluar(self, solucion_obj):
        """evaluar(Item[]) Suma el peso y valor (evalua la funcion objetivo) de una combinacion (lista) de Items

        Parameters
        ----------
        solucion_obj : Item list
            Lista de objetos de la clase Item

        Returns
        -------
        dict
            Diccionario con las siguientes clave-valor:
                valor : suma de todos los valores de la lista de items
                peso  : suma de todos los pesos de la lista de items
        """
        sum_valor = 0
        sum_peso = 0
        for x in solucion_obj:
            sum_valor += x.valor_item
            sum_peso += x.peso_item
        return {"valor": sum_valor, "peso": sum_peso}

    def random_neighbor(self, solucion_dict):
        """random_neighbor({"binary": [1,0,1,...,n in {1,0}], "obj": Item[]})
        Esta funcion mete o saca de manera aleatoria un item de la mochila, cada
        return de esta funcion se le conoce como 'vecino'. Sacar implica colocar
        un 0 en el arreglo binario en la posicion del item aleatorio seleccionado y meter es un 1.
        Una vez se metio/saco un item, el arreglo resultante se clona a si mismo para evitar
        la mutacion (ver que son objetos mutables en python) no deseable fuera de la funcion.

        Parameters
        ----------
        solucion_dict : dict con las siguientes clave-valor
            binary : Lista [1,0,1,...,n in {1,0}]
            obj    : Lista de objetos de la clase Item
            Ambos arreglos deben ser del mismo tamaño (ya que binary se forma a partir de obj)

        Returns
        -------
        dict
            Diccionario con las siguientes clave-valor:
                binary : Lista [1,0,1,...,n in {1,0}] 'mutada' distinta a la lista de entrada
                obj    : Lista 'mutada' de objetos de la clase Item formada a partir del 'binary' mutado
                         1 = se coloca dentro de la lista, 0 = no se coloca en la lista
            La salida tiene la misma estructura que la entrada, sin embargo la salida es
            una copia superficial (copia en terminos de progrmacion) de la mutacion de la entrada
        """
        r = copy.deepcopy(solucion_dict)
        bit_flip = randint(len(self.items), size=1)[0]
        # Meter = 1, Sacar = 0
        bit_flip_valor = solucion_dict['binary'][bit_flip]
        solucion_dict['binary'][bit_flip] = 1 ^ bit_flip_valor
        peso = sum(v.peso_item for i,v in enumerate(self.items) if solucion_dict['binary'][i]==1)
        if peso <= self.max_peso:
            solucion_dict['obj'] = list((self.items[i] for i,v in enumerate(solucion_dict['binary']) if v==1))
            return solucion_dict
        else:
            return r

    def initial_solution(self, only_binary = True):
        """initial_solution() Esta funcion genera un solucion inicial de manera aleatoria para el problema.
        La sulucion inicial consta de cualquier combinacion(lista) de items que no superen
        el peso maximo de la mochila. Es aqui donde se crea la relacion entre la solucion
        binaria y la solucion de items, por esta razon las posiciones(index) de cada valor
        en la solucion binaria corresponde a la la misma posicion del item en su respectiva lista.

        Parameters
        ----------
        Ninugno. Toma las propiedades de la clase (Item[], peso_maximo)

        Returns
        -------
        dict
            Diccionario con las siguientes clave-valor:
                binary : Lista [1,0,1,...,n in {1,0}] como combinacion inicial que no supera el peso maximo de la mochila (puede que sea la solucion final)
                obj    : Lista de objetos de la clase Item formada a partir del cualquier combinacion que no supera el peso maximo de la mochila (puede que sea la solucion final)
        """
        initial_solucion = []
        binary_solucion = [0 for x in self.items]
        suma_peso = 0

        while suma_peso <= self.max_peso:
            index_item = randint(len(self.items), size=1)[0]
            suma_peso += sum(v.peso_item for i,v in enumerate(self.items) if binary_solucion[i]==1 ) + self.items[index_item].peso_item
            if suma_peso <= self.max_peso:
                binary_solucion[index_item] = 1
        initial_solucion = list((self.items[i] for i,v in enumerate(binary_solucion) if v==1))
        if only_binary:
            return binary_solucion
        else:
            return {"binary": binary_solucion, "obj": initial_solucion}

    def __str__(self):
        s = self.solve_sa(True)
        return """Solucion binaria: {}\nLos items que resuelven el problema son: {}\nTotal de Items: {}\ncon un peso total de: {}\ny un valor maximo de: {}""".format(s["solucion_binary"],
                                    [x.id for x in s["solucion_objs"]],
                                    len(s["solucion_objs"]),s['peso'],s['valor'])

if __name__ == '__main__':
    #TODO Documentar esta seccion (ejecucion por consola)
    import getopt
    import sys

    items = []

    items_valores = []
    items_pesos   = []
    peso_mochila  = 0
    max_iter      = 0
    random        = True
    terminal      = False

    try:
        options, remainder = getopt.getopt(
            sys.argv[1:],
            'v:p:m:i:r:t',
            ['valores=',
            'pesos=',
            'mochila=',
            'iteraciones=',
            'random=',
            'terminal=',
            ]
        )
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)

    for opt, arg in options:
        if opt in ('-v', '--valores'):
            items_valores = arg.split(",")
            random = False
        elif opt in ('-p', '--pesos'):
            items_pesos = arg.split(",")
        elif opt in ('-m', '--mochila'):
            peso_mochila = int(arg)
        elif opt in ('-i', '--iteraciones'):
            max_iter = int(arg)
        elif opt in ('-r', '--random'):
            random = True
        elif opt in ('-t', '--terminal'):
            terminal = True

    try:
        if random:
            for x in range(100):
                aux = Item(x,randint(1, 100),randint(1, 100))
                items.append(aux)
            peso_mochila = 100
            max_iter = 1000
        else:
            for i,(v,p) in enumerate(zip(items_valores,items_pesos)):
                aux = Item(i,int(v),int(p))
                items.append(aux)
    except Exception as e:
        print("Oh no! this isn't the way, rompio el programa, contacte al desarrollador")
        print("ERROR:", e)
        sys.exit(1)


    m = Mochila(peso_mochila, items, max_iter)
    if terminal:
        print(m)
    else:
        try:
            print(m.solve_rmhc(False))
        except Exception as e:
            print(e)
