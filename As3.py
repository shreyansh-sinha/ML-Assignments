import numpy as np
import random

# Initial parameters
# length of binary vector in generation = 6
# population size = 10
# parent selection = roulette wheel selection
# mutation probability = 0.01
# survival selection = Replace 20% worse solutions in children (Offspring) Population with 20% best solution in the parent population.

l = 6
num_population = 10
mutation_prob = 0.01

def define_population():
  pop_list = []
  pop_list = np.random.randint(low=0, high=63, size=10)

  return pop_list

def fitness_value(pop_list):
  fitness_list = []

  for pop in pop_list:
    fitness_list.append(pop**3 + 9)
  
  return fitness_list


def get_prob(fitness_list):
  prob_list = []

  for pop in fitness_list:
    prob_list.append(pop/sum(fitness_list))
  
  return prob_list


# based on roulette wheel selection
def get_Parents(binary_list, prob_list):
  return np.random.choice(binary_list, p = prob_list)


def crossover(p1, p2, n):
  c1 = []
  c2 = []

  for i in range(6):
    if i < n:
      c1.append(p1[i])
      c2.append(p2[i])
    else:
      c1.append(p2[i])
      c2.append(p1[i])
  return c1, c2


def get_children(list1, list2):
  n = np.random.randint(1, 6)
  
  for i in range(5):
    parent1 = list1[i]
    parent2 = list2[i]

  child1, child2 = crossover(parent1, parent2, n)
  return child1, child2


def mutation(children_list, mutation_prob):

  for child in children_list:
    for idx in range(6):
      rand_num = random.uniform(0, 0.02)

      if rand_num < mutation_prob:
        if child[idx] == '1':
          child[idx] = '0'
        else:
          child[idx] = '1'

  return children_list


def survivor_selection(parent, child):
  temp_list1 = []
  temp_list2 = []
  
  min_child1 = 100
  min_child2 = 100
  for ele in child:
    temp_list1.append(int(ele, 2))

  for ele in parent:
    temp_list2.append(int(ele, 2))

  # print(temp_list1)
  # print(temp_list2)

  for ele in temp_list1:
    if ele < min_child1:
      min_child1 = ele
  
  temp_list1.remove(min_child1)

  for ele in temp_list1:
    if ele < min_child2:
      min_child2 = ele
  
  temp_list1.remove(min_child2)

  max_child1 = 0
  max_child2 = 0

  for ele in temp_list2:
    if ele > max_child1:
      max_child1 = ele

  for ele in temp_list2:
    if ele > max_child2:
      max_child2 = ele
  
  temp_list1.append(max_child1)
  temp_list1.append(max_child2)

  # print(temp_list1)

  return temp_list1
  

itr = 0

next_gen_children = []
pop_list = []

mx = 0

while itr < 1000:

  if itr == 0:
    pop_list = define_population()
  else:
    pop_list = next_gen_children
    
  binary_list = []

  for pop in pop_list:
    binary_list.append(bin(pop)[2:].zfill(6))
  
  fitness_list = fitness_value(pop_list)

  prob_list = get_prob(fitness_list)

  parents_list = []
  
  for i in range(10):
    parent = get_Parents(binary_list, prob_list)
    parents_list.append(parent)
  
  crossover_list1 = []
  crossover_list2 = []

  for i in range(10):
    if i < 5:
      crossover_list1.append(parents_list[i])
    else:
      crossover_list2.append(parents_list[i])
    
  children_list = []
  
  for i in range(5):
    child1, child2 = get_children(crossover_list1, crossover_list2)

    children_list.append(child1)
    children_list.append(child2)
  
  next_gen_child_list = []
  next_gen_child_list = mutation(children_list, mutation_prob)

  # print(parents_list)
  
  new_child_list = []

  for child in next_gen_child_list:
    idx = 5
    s = 0
    while idx >= 0:
      s += pow(2, int(child[idx]))
      idx = idx-1
    
    new_child_list.append((bin(s)[2:].zfill(6)))

  # print(new_child_list)

  next_gen_children = survivor_selection(parents_list, new_child_list)

  mx = max(mx, max(next_gen_children))
  itr = itr + 1

print(mx)


