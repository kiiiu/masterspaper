# -*- coding: utf-8 -*-
import math
import numpy
import random

import graph_tool.all as gt
from scipy.optimize import linprog, minimize

# Constants below
NODES_NUMBER = 5

X_MAX = 5000.0  # m
Y_MAX = 5000.0  # m

POWER = 0.5  # Watt
NODES_POWER_VECTOR = [POWER] * NODES_NUMBER
RSSI_MIN = -90  # dBm

FREQUENCY_LOW_BORDER = 2.4 * 10**9  # Hz
FREQUENCY_BANDWIDTH = 20 * 10**6  # Hz

TEMPERATURE = 300  # K
K = 1.38 * 10**(-23)  # J/K
NOISE_POWER = FREQUENCY_BANDWIDTH * K * TEMPERATURE

REQUIRED_DATARATE_BETWEEN_NODES = 1*10**6  # bit/s

# Computations below

# Topology initiation
def initiate_topology():
  nodes_coordinates = [(X_MAX*random.random(), Y_MAX*random.random()) for node in range(NODES_NUMBER)]
  return nodes_coordinates


def calculate_nodes_distances(coordinates):
  nodes_distances = [None] * NODES_NUMBER
  for node in range(NODES_NUMBER):
    distances_to_node = [math.sqrt((coordinates[node][0] - x) ** 2 +
               (coordinates[node][1] - y) ** 2) for x, y in coordinates]
    nodes_distances[node] = distances_to_node
  return nodes_distances


def attenuated_power(power, distance):
  carrier_frequency = FREQUENCY_LOW_BORDER + FREQUENCY_BANDWIDTH / 2.0
  antenna_gain = 1

  # Friis transmission equation (free-space loss)
  if distance > 0:
    received_power = power*antenna_gain**2 * ((3*10**8/carrier_frequency) * (1/float(4*math.pi*distance)))**2
  else:
    return power
  if received_power < power:
    return received_power
  else:
    return power


def calculate_received_power(power_vector, distance_matrix):
  received_power = [[0 for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
    received_power[transmitter][receiver] = attenuated_power(power_vector[transmitter], distance_matrix[transmitter][receiver])
  return received_power


def find_adjacency(received_power, min_dbm_level):
  adjacency_matrix = [[None for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if 10 * math.log10(received_power[transmitter][receiver] / 0.001) >= min_dbm_level and transmitter != receiver:
        adjacency_matrix[transmitter][receiver] = 1
      else:
        adjacency_matrix[transmitter][receiver] = 0
  return adjacency_matrix

# Available routes finding
def calculate_routes_number(nodes_routes):
  total_routes_number = 0
  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      total_routes_number += len(nodes_routes[starting_node][end_node])
  return total_routes_number


def find_all_routes(adjacency_matrix):
  def find_recursive_routes(sender_node, end_node):
    nodes_routes = []
    current_route = [sender_node]

    def one_node_addition(current_route):
      for node in range(NODES_NUMBER):
        if adjacency_matrix[current_route[-1]][node]:
          if node == end_node:
            nodes_routes.append(current_route[:] + [node])
          elif node not in current_route and len(current_route) < NODES_NUMBER-1:
            one_node_addition(current_route[:]+[node])

    one_node_addition(current_route)  # Recursive, use with caution
    nodes_routes.sort(key=len)
    return nodes_routes

  print('Finding routes...')
  nodes_routes = [[[] for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  total_routes_number = 0

  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      if starting_node != end_node:
        nodes_routes[starting_node][end_node] = find_recursive_routes(starting_node, end_node)
        total_routes_number += len(nodes_routes[starting_node][end_node])
        if not nodes_routes[starting_node][end_node]:
          print('No routes between ' + str(starting_node) + ' and ' + str(end_node) + ' node')

  print('Total routes number - ' + str(total_routes_number))
  return nodes_routes

#Sort routes by edges of graph they going
def find_routes_through_links(nodes_routes):
  routes_through_links = [[[] for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      for route in nodes_routes[starting_node][end_node]:
        route_index = nodes_routes[starting_node][end_node].index(route)
        hops_number = len(route) - 1
        for hop in range(hops_number):
          routes_through_links[route[hop]][route[hop + 1]].append((starting_node, end_node, route_index))
  return routes_through_links

# Datarate requred by every node
def calculate_datarates_for_all(nodes_routes):
  datarates_matrix = [[0 for starting_node in range(NODES_NUMBER)] for end_node in range(NODES_NUMBER)]
  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      if nodes_routes[starting_node][end_node]:
        datarates_matrix[starting_node][end_node] = REQUIRED_DATARATE_BETWEEN_NODES
  return datarates_matrix

# Initial solution
# Frequency fragmentation between nodes
def find_dependent_links(matrix_of_adjacency):
  dependent_links = []
  for node in range(0, NODES_NUMBER):
    links = []
    for adjacent_node in range(0, NODES_NUMBER):
      if matrix_of_adjacency[node][adjacent_node]:
        links.append((node, adjacent_node))
        links.append((adjacent_node, node))
        for adjacent_to_adjacent_node in range(0, NODES_NUMBER):
          if matrix_of_adjacency[adjacent_node] [adjacent_to_adjacent_node]:
            links.append((adjacent_node, adjacent_to_adjacent_node))
            links.append((adjacent_to_adjacent_node, adjacent_node))
            for adjacent_to_adjacent_to_adjacent_node in range(0, NODES_NUMBER):
              if matrix_of_adjacency[adjacent_to_adjacent_node] [adjacent_to_adjacent_to_adjacent_node]:
                links.append((adjacent_to_adjacent_node, adjacent_to_adjacent_to_adjacent_node))
    dependent_links.append(frozenset(links))
  return set(dependent_links)


def find_links_capacity(nodes_frequency_bands, received_power):
  frequency_matrix = numpy.array(nodes_frequency_bands).reshape((NODES_NUMBER, NODES_NUMBER))
  links_capacity = [[0 for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if transmitter != receiver:
        f = frequency_matrix.item((transmitter, receiver))
        snr = received_power[transmitter][receiver]/NOISE_POWER
        links_capacity[transmitter][receiver] = f * math.log(1+snr, 2)
  return links_capacity


def find_routes_capacity(links_capacity, routes):
  def calculate_route_capacity(route):
    if route:
      hops_number = len(route) - 1
      min = None
      for hop in range(hops_number):
        link_capacity = links_capacity[route[hop]][route[hop+1]]
        if not min or link_capacity < min:
          min = link_capacity
      return min
    else:
      return None

  routes_capacities = []
  for starting_node in range(NODES_NUMBER):
    routes_capacities.append([])
    for end_node in range(NODES_NUMBER):
      routes_capacities[starting_node].append([])
      for route in routes[starting_node][end_node]:
        routes_capacities[starting_node][end_node].append (calculate_route_capacity(route))

  return routes_capacities


def convert_vector_to_datarate_freq(vector, nodes_routes):
  nodes_passed = 0
  routes_datarate = []
  for starting_node in range(NODES_NUMBER):
    routes_datarate.append([])
    for end_node in range(NODES_NUMBER):
      routes_datarate[starting_node].append([])
      for route in nodes_routes[starting_node][end_node]:
        routes_datarate[starting_node][end_node].append (vector.item(nodes_passed))
        nodes_passed += 1
  frequencies = []
  for transmitter in range(NODES_NUMBER):
    frequencies.append([])
    for receiver in range(NODES_NUMBER):
      if transmitter != receiver:
        frequencies[transmitter].append (vector.item(nodes_passed))
        nodes_passed += 1
      else:
        frequencies[transmitter].append(0)
  return routes_datarate, frequencies


def find_initial_solution(nodes_routes, dependent_links, received_power, vector_output=False):
  total_routes_number = calculate_routes_number(nodes_routes)
  routes_through_links = find_routes_through_links(nodes_routes)
  datarates_matrix = calculate_datarates_for_all(nodes_routes)

  A_ub = []
  b_ub = []
  max_load = 1 - 10**(-3)
  c = [0] * total_routes_number
  A_eq = []
  b_eq = []

  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if transmitter != receiver:
        c.append(-math.log(1 + received_power[transmitter][receiver] / NOISE_POWER, 2))

  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if routes_through_links[transmitter][receiver]:
        A_line = []
        for starting_node in range(NODES_NUMBER):
          for end_node in range(NODES_NUMBER):
            for route in nodes_routes[starting_node][end_node]:
              if (starting_node, end_node, nodes_routes[starting_node] [end_node].index(route)) in routes_through_links[transmitter][receiver]:
                A_line.append(1)
              else:
                A_line.append(0)
        if A_line:
          for tx in range(0, NODES_NUMBER):
            for rx in range(0, NODES_NUMBER):
              if tx != rx:
                if transmitter == tx and receiver == rx:
                  A_line.append(-math.log(1 + received_power[tx][rx]/ NOISE_POWER, 2) * max_load)
                else:
                  A_line.append(0)
          A_ub.append(A_line)
          b_ub.append(0)

  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      A_line = []
      if datarates_matrix[starting_node][end_node]:
        for first_node in range(NODES_NUMBER):
          for last_node in range(NODES_NUMBER):
            for route in nodes_routes[first_node][last_node]:
              if starting_node == first_node and end_node == last_node:
                A_line.append(1)
              else:
                A_line.append(0)
        A_line += [0 for tx in range(NODES_NUMBER) for rx in range(NODES_NUMBER) if tx != rx]
        A_eq.append(A_line)
        b_eq.append(datarates_matrix[starting_node][end_node])

  for dependent_links_set in dependent_links:
    A_line = [0] * total_routes_number
    for transmitter in range(NODES_NUMBER):
      for receiver in range(NODES_NUMBER):
        if transmitter != receiver:
          if (transmitter, receiver) in dependent_links_set:
            A_line.append(1)
          else:
            A_line.append(0)
    A_ub.append(A_line)
    b_ub.append(FREQUENCY_BANDWIDTH)

  out = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

  if vector_output:
    return out.x
  else:
    return convert_vector_to_datarate_freq(out.x, nodes_routes)


def find_links_datarate(routes_datarate, nodes_routes): # find -> convert to ...
  links_datarate = [[0 for receiver in range(NODES_NUMBER)] for transmitter in range(NODES_NUMBER)]
  for starting_node in range(NODES_NUMBER):
    for end_node in range(NODES_NUMBER):
      for route in nodes_routes[starting_node][end_node]:
        hops_number = len(route) - 1
        for hop in range(hops_number):
          links_datarate[route[hop]][route[hop+1]] += routes_datarate[starting_node][end_node] [nodes_routes[starting_node] [end_node].index(route)]
  return links_datarate


def target_function(routes_and_freq_vector, nodes_routes, received_power):
  function = 0

  nodes_passed = 0
  routes_datarate = []
  for starting_node in range(NODES_NUMBER):
    routes_datarate.append([])
    for end_node in range(NODES_NUMBER):
      routes_datarate[starting_node].append([])
      for route in nodes_routes[starting_node][end_node]:
        routes_datarate[starting_node][end_node].append (routes_and_freq_vector.item(nodes_passed))
        nodes_passed += 1
  frequencies = []
  for transmitter in range(NODES_NUMBER):
    frequencies.append([])
    for receiver in range(NODES_NUMBER):
      if transmitter != receiver:
        frequencies[transmitter].append (routes_and_freq_vector.item (nodes_passed))
        nodes_passed += 1
      else:
        frequencies[transmitter].append(0)

  links_datarate = find_links_datarate(routes_datarate, nodes_routes)
  links_capacity = find_links_capacity(frequencies, received_power)

  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if links_datarate[transmitter][receiver] > 1.1:
        load = links_datarate[transmitter][receiver] / links_capacity[transmitter][receiver]
        if load < 1:
          function += load/(1-load)
        else:
          function = float('inf')
  return function


def find_optimized_solution(initial_vector_solution, nodes_routes, received_power, dependent_links, vector_output=False):
  total_routes_number = calculate_routes_number(nodes_routes)
  routes_through_links = find_routes_through_links(nodes_routes)
  datarates_matrix = calculate_datarates_for_all(nodes_routes)

  bounds = []

  for route in range(total_routes_number):
    bounds.append((0, None))
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if transmitter != receiver:
        if not routes_through_links[transmitter][receiver]:
          bounds.append((0, 0))
        else:
          bounds.append((0, FREQUENCY_BANDWIDTH))

  def generate_constraints():
    constraints = []
    max_load = 1 - 10**(-10)

    class Constraint:
      def __init__(self, A_line, const=0):
        self.A_line = A_line
        self.const = const

      def __call__(self, vector):
        function = 0
        for variable_position in range(vector.size):
          function += self.A_line[variable_position] * vector.item (variable_position)
        return function - self.const

    for transmitter in range(NODES_NUMBER):
      for receiver in range(NODES_NUMBER):
        if routes_through_links[transmitter][receiver]:
          A_line = []
          for starting_node in range(NODES_NUMBER):
            for end_node in range(NODES_NUMBER):
              for route in nodes_routes[starting_node][end_node]:
                if (starting_node, end_node, nodes_routes[starting_node] [end_node].index(route)) in routes_through_links[transmitter][receiver]:
                  A_line.append(-1)
                else:
                  A_line.append(0)
          if A_line:
            for tx in range(0, NODES_NUMBER):
              for rx in range(0, NODES_NUMBER):
                if tx != rx:
                  if transmitter == tx and receiver == rx:
                    A_line.append(math.log(1 + received_power[tx][rx] / NOISE_POWER, 2) * max_load)
                  else:
                    A_line.append(0)
            constraints.append({'type': 'ineq', 'fun': Constraint(A_line)})

    for starting_node in range(NODES_NUMBER):
      for end_node in range(NODES_NUMBER):
        A_line = []
        if datarates_matrix[starting_node][end_node]:
          for first_node in range(NODES_NUMBER):
            for last_node in range(NODES_NUMBER):
              for route in nodes_routes[first_node][last_node]:
                if starting_node == first_node and end_node == last_node:
                  A_line.append(1)
                else:
                  A_line.append(0)
          A_line += [0 for tx in range(NODES_NUMBER) for rx in range(NODES_NUMBER) if tx != rx]
          constraints.append({'type': 'eq', 'fun': Constraint(A_line, datarates_matrix[starting_node][end_node])})

    for dependent_links_set in dependent_links:
      A_line = [0] * total_routes_number
      for transmitter in range(NODES_NUMBER):
        for receiver in range(NODES_NUMBER):
          if transmitter != receiver:
            if (transmitter, receiver) in dependent_links_set:
              A_line.append(1)
            else:
              A_line.append(0)
      constraints.append({'type': 'ineq', 'fun': Constraint(A_line, -FREQUENCY_BANDWIDTH)})

    return constraints

  constraints = generate_constraints()

  x0 = numpy.array([int(round(initial_vector_solution.item(value))) for value in range(initial_vector_solution.size)])

  def check_for_correctness(x0):
    routes_datarate, frequency = convert_vector_to_datarate_freq(x0, nodes_routes)
    links_datarate = find_links_datarate(routes_datarate, nodes_routes)
    links_capacity = find_links_capacity(frequency, received_power)
    for transmitter in range(NODES_NUMBER):
      for receiver in range(NODES_NUMBER):
        if not links_datarate[transmitter][receiver] <= links_capacity[transmitter][receiver]:
          print('Transmitter - ' + str(transmitter) + '; receiver - ' + str(receiver))
          print('Datarate - ' + str(links_datarate[transmitter][receiver]) + '; capacity - ' + str(links_capacity[transmitter][receiver]))
          return False
        elif int(sum(routes_datarate[transmitter][receiver])) != int(datarates_matrix[transmitter][receiver]):
          print('Transmitter - ' + str(transmitter) + '; receiver - ' + str(receiver))
          print('Routes - ' + str(sum(nodes_routes[transmitter][receiver])) + '; needed datarate - ' + str(datarates_matrix[transmitter][receiver]))
          return False
    return True

  local_target_function = lambda solution_vector: target_function(solution_vector, nodes_routes, received_power)

  if check_for_correctness(x0):
    print('Initial solution was correct')
  else:
    print("Initial solution wasn't correct")

  print('Function value of initial solution - ' + str(local_target_function(x0)))

  for bound in bounds:
    if bound[0] and not x0[bounds.index(bound)] > bound[0]:
        print('''Doesn't match bounds''')
    if bound[1] and not x0[bounds.index(bound)] < bound[1]:
        print('''Doesn't match bounds''')

  for constraint in constraints:
    if constraint['type'] == 'eq':
      if constraint['fun'](x0) != 0:
        print('''Doesn't match to ''' + str(constraints.index(constraint)) + ' constraint')
        print('Function value - ' + str(constraint['fun'](x0)))
    elif constraint['type'] == 'ineq':
      if constraint['fun'](x0) < 0:
        print('''Doesn't match to ''' + str(constraints.index(constraint)) + ' constraint')
        print('Function value - ' + str(constraint['fun'](x0)))

  optimized_solution = minimize(local_target_function, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp':True})

  if vector_output:
    return optimized_solution.x
  else:
    return convert_vector_to_datarate_freq(optimized_solution.x, nodes_routes)


def visualize_graph(nodes_coords, datarate, matrix_of_adjacency, edge_color=None):
  graph = gt.Graph(directed=True)

  vertex_list = [graph.add_vertex() for node in range(NODES_NUMBER)]

  maximal_datarate = 0
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if datarate[transmitter][receiver] > maximal_datarate:
        maximal_datarate = datarate[transmitter][receiver]

  edges_width = []
  for transmitter in range(NODES_NUMBER):
    for receiver in range(NODES_NUMBER):
      if matrix_of_adjacency[transmitter][receiver]:
        graph.add_edge(transmitter, receiver)
        edges_width.append(10 * datarate[transmitter][receiver]/maximal_datarate)

  pos = graph.new_vertex_property("vector<float>", vals=numpy.array(nodes_coords))
  from time import time
  if not edge_color:
    gt.graph_draw(graph,
            vertex_text=graph.vertex_index,
            pos=pos,
            vertex_font_size=20,
            vertex_halo_size=30,
            vertex_size=60,
            output_size=(800, 700),
            edge_pen_width=graph.new_edge_property("double", vals=edges_width),
            output="Results/" + str(time()) + ".png")
  else:
    gt.graph_draw(graph,
            vertex_text=graph.vertex_index,
            pos=pos,
            vertex_font_size=20,
            vertex_halo_size=30,
            vertex_size=60,
            output_size=(800, 700),
            edge_pen_width=graph.new_edge_property("double", vals=edges_width),
            edge_color=edge_color,
            output="Results/" + str(time())+".png")

# And action!

# Initiation
coordinates = initiate_topology()
distances_matrix = calculate_nodes_distances(coordinates)
received_power_matrix = calculate_received_power(NODES_POWER_VECTOR, distances_matrix)
adjacency_matrix = find_adjacency(received_power_matrix, RSSI_MIN)
dependent_links_sets = find_dependent_links(adjacency_matrix)
routes = find_all_routes(adjacency_matrix)

# Start finding
initial_solution = find_initial_solution(routes, dependent_links_sets, received_power_matrix, True)
try:
  has_solution = not numpy.isnan(initial_solution.all())
except AttributeError:
  has_solution = False
if not has_solution:  # Is solution "None"? (numpy nan)
  print('No solution for this configuration')
else:
  initial_routes_datarate, initial_freq = convert_vector_to_datarate_freq(initial_solution, routes)
  initial_links_datarate = find_links_datarate(initial_routes_datarate, routes)
  initial_links_capacity = find_links_capacity(initial_freq, received_power_matrix)

  visualize_graph(coordinates, initial_links_datarate, adjacency_matrix, [.78, .2, .2, 1])
  visualize_graph(coordinates, initial_links_capacity, adjacency_matrix, [0, .75, .75, .25])

  # Scipy optimization

  optimized_solution = find_optimized_solution(initial_solution, routes, received_power_matrix, dependent_links_sets, True)
  optimized_routes_datarate, optimized_freq = convert_vector_to_datarate_freq(optimized_solution, routes)
  optimized_links_datarate = find_links_datarate(optimized_routes_datarate, routes)
  optimized_links_capacity = find_links_capacity(optimized_freq, received_power_matrix)

  visualize_graph(coordinates, optimized_links_datarate, adjacency_matrix, [.78, .2, .2, 1])
  visualize_graph(coordinates, optimized_links_capacity, adjacency_matrix, [0, .75, .75, .25])
