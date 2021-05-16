###########################
# 6.0002 Problem Set 1a: Space Cows
# Name:
# Collaborators:
# Time:

from ps1_partition import get_partitions
from operator import itemgetter
import time

# ================================
# Part A: Transporting Space Cows
# ================================


# Problem 1
def load_cows(filename):
    """
    Read the contents of the given file.  Assumes the file contents contain
    data in the form of comma-separated cow name, weight pairs, and return a
    dictionary containing cow names as keys and corresponding weights as values.

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a dictionary of cow name (string), weight (int) pairs
    """
    cows_data = {}

    file = open(filename, 'r')

    for line in file:
        name, weight = line.split(',')
        cows_data[name] = int(weight.rstrip())

    file.close()

    return cows_data


# Problem 2
def greedy_cow_transport(cows, limit=10):
    """
    Uses a greedy heuristic to determine an allocation of cows that attempts to
    minimize the number of spaceship trips needed to transport all the cows. The
    returned allocation of cows may or may not be optimal.
    The greedy heuristic should follow the following method:

    1. As long as the current trip can fit another cow, add the largest cow that will fit
        to the trip
    2. Once the trip is full, begin a new trip to transport the remaining cows

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)

    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    # list of tuples in form of (name, weight) sorted by weight
    # from bigger to smaller
    cows_by_weight = sorted(cows.items(), key=lambda x: x[1], reverse=True)
    all_trips = []

    while len(cows_by_weight) > 0:
        current_trip = []
        current_weight = 0

        # looping the list while modifying it will lead to unexpected results
        # hence the copy
        for cow in cows_by_weight.copy():
            name, weight = cow

            if current_weight + weight <= limit:
                current_trip.append(name)
                current_weight += weight
                cows_by_weight.remove(cow)

        all_trips.append(current_trip)

    return all_trips


# Problem 3
def brute_force_cow_transport(cows, limit=10):
    """
    Finds the allocation of cows that minimizes the number of spaceship trips
    via brute force.  The brute force algorithm should follow the following method:

    1. Enumerate all possible ways that the cows can be divided into separate trips
        Use the given get_partitions function in ps1_partition.py to help you!
    2. Select the allocation that minimizes the number of trips without making any trip
        that does not obey the weight limitation

    Does not mutate the given dictionary of cows.

    Parameters:
    cows - a dictionary of name (string), weight (int) pairs
    limit - weight limit of the spaceship (an int)

    Returns:
    A list of lists, with each inner list containing the names of cows
    transported on a particular trip and the overall list containing all the
    trips
    """
    best_trips = []

    for partition in get_partitions(cows.keys()):
        # capacities for current partition
        all_weights = []

        for trip in partition:
            # individual capacity for each element of partition
            trip_weight = sum(cows[name] for name in trip)
            all_weights.append(trip_weight)

        # each capacity is within a limit
        if all(weight <= limit for weight in all_weights):
            # and current partition is better than than previosuly saved one
            if len(partition) <= len(best_trips) or len(best_trips) == 0:
                # redeclare new best
                best_trips = partition

    return best_trips


# Problem 4
def compare_cow_transport_algorithms():
    """
    Using the data from ps1_cow_data.txt and the specified weight limit, run your
    greedy_cow_transport and brute_force_cow_transport functions here. Use the
    default weight limits of 10 for both greedy_cow_transport and
    brute_force_cow_transport.

    Print out the number of trips returned by each method, and how long each
    method takes to run in seconds.

    Returns:
    Does not return anything.
    """
    cows = load_cows('ps1_cow_data.txt')

    start = time.time()
    greedy_cow_transport(cows)
    end = time.time()
    print(greedy_cow_transport.__name__, 'took', end - start, 'seconds to run')

    start = time.time()
    brute_force_cow_transport(cows)
    end = time.time()
    print(brute_force_cow_transport.__name__,
          'took', end - start, 'seconds to run')


if __name__ == '__main__':
    compare_cow_transport_algorithms()
