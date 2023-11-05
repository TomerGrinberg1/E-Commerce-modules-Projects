import numpy as np
import networkx as nx
import random
import pandas as pd

# CHANGE ################################
# influencers = [0, 1, 2, 3, 4]
from scipy.signal import firls


def test1():
    costs = pd.read_csv("costs.csv")
    neighbors = pd.read_csv("chic_choc_data.csv")
    top_25_most_neighbors = neighbors['user'].value_counts().nlargest(25)
    costs = costs.sort_values('cost', ascending=True)
    influ = []
    merged_df = pd.merge(neighbors, costs, on='user')
    variable = 0.6
    cost_list = costs["user"].to_list()
    while len(influ) < 5:
        for i in range(len(top_25_most_neighbors)):
            if len(influ) < 5:
                user = top_25_most_neighbors.keys()[i]
                if user not in cost_list:
                    influ.append(user)
                else:
                    if costs.loc[costs["user"] == user, "cost"].values[0] < variable * top_25_most_neighbors.values[i]:
                        influ.append(user)

    print(influ)
    return influ


# CHANGE ################################

p = 0.01
chic_choc_path = 'chic_choc_data.csv'
cost_path = 'costs.csv'


def create_graph(edges_path: str) -> nx.Graph:
    """
    Creates the Chic Choc social network
    :param edges_path: A csv file that contains information obout "friendships" in the network
    :return: The Chic Choc social network
    """
    edges = pd.read_csv(edges_path).to_numpy()
    net = nx.Graph()
    net.add_edges_from(edges)
    return net


def change_network(net: nx.Graph) -> nx.Graph:
    """
    Gets the network at staged t and returns the network at stage t+1 (stochastic)
    :param net: The network at staged t
    :return: The network at stage t+1
    """
    edges_to_add = []
    for user1 in sorted(net.nodes):
        for user2 in sorted(net.nodes, reverse=True):
            if user1 == user2:
                break
            if (user1, user2) not in net.edges:
                neighborhood_size = len(set(net.neighbors(user1)).intersection(set(net.neighbors(user2))))
                prob = 1 - ((1 - p) ** (np.log(neighborhood_size))) if neighborhood_size > 0 else 0  # #################
                if prob >= random.uniform(0, 1):
                    edges_to_add.append((user1, user2))
    net.add_edges_from(edges_to_add)
    return net


def buy_products(net: nx.Graph, purchased: set) -> set:
    """
    Gets the network at the beginning of stage t and simulates a purchase round
    :param net: The network at stage t
    :param purchased: All the users who bought a doll up to and including stage t-1
    :return: All the users who bought a doll up to and including stage t
    """
    new_purchases = set()
    for user in net.nodes:
        neighborhood = set(net.neighbors(user))
        b = len(neighborhood.intersection(purchased))
        n = len(neighborhood)
        prob = b / n
        if prob >= random.uniform(0, 1):
            new_purchases.add(user)

    return new_purchases.union(purchased)


def get_influencers_cost(cost_path: str, influencers: list) -> int:
    """
    Returns the cost of the influencers you chose
    :param cost_path: A csv file containing the information about the costs
    :param influencers: A list of your influencers
    :return: Sum of costs
    """
    costs = pd.read_csv(cost_path)
    return sum(
        [costs[costs['user'] == influencer]['cost'].item() if influencer in list(costs['user']) else 0 for influencer in
         influencers])


if __name__ == '__main__':

    print("STARTING")

    chic_choc_network = create_graph(chic_choc_path)


    def test2():  # centrality
        # betweenness = nx.betweenness_centrality(chic_choc_network)
        # closeness = nx.closeness_centrality(chic_choc_network)
        degree = nx.degree_centrality(chic_choc_network)
        # Eigenvector=nx.eigenvector_centrality(chic_choc_network)

        # harmonic=nx.harmonic_centrality(chic_choc_network)
        # pagerank = nx.pagerank(chic_choc_network)

        # load=nx.load_centrality(chic_choc_network)
        # katz=nx.katz_centrality(chic_choc_network)
        costs = pd.read_csv("costs.csv")
        cost_list = costs["user"].to_list()

        sorted_dict = dict(sorted(degree.items(), key=lambda x: x[1], reverse=True))
        influ = []
        for i in range(len(sorted_dict.keys())):
            user = list(sorted_dict.keys())[i]

            if user in cost_list:
                if costs.loc[costs["user"] == user, "cost"].values[0] < 1000:
                    if len(influ) < 5:
                        influ.append(user)
            else:
                if len(influ) < 5:
                    influ.append(user)

        print(influ)
        return influ


    def combine(G):
        neighbors = pd.read_csv("chic_choc_data.csv")

        page_scores = nx.pagerank(G)
        costs = pd.read_csv("costs.csv")
        cost_list = costs["user"].to_list()
        # Compute the degree scores for all nodes
        degree_scores = dict(G.degree())
        neighbors_count = neighbors['user'].value_counts()

        # Compute the average score for the node
        average_scores = {node: (page_scores[node] * 0.9 + degree_scores[node] * 0.1) for node in G if (
                node in cost_list and costs.loc[costs["user"] == node, "cost"].values[
            0] < 0.6 * neighbors_count[node])}

        # Print the average score

        # Compute the best average score across all nodes
        top_nodes = sorted(average_scores, key=average_scores.get, reverse=True)[:5]

        # Print the best average score
        print(top_nodes)
        return top_nodes


    def test3(graph):
        num_walks = 1000
        walk_length = 5
        costs = pd.read_csv("costs.csv")
        cost_list = costs["user"].to_list()
        neighbors = pd.read_csv("chic_choc_data.csv")
        neighbors_count = neighbors['user'].value_counts()

        visit_counts = {n: 0 for n in graph.nodes}
        for _ in range(num_walks):
            for n in graph.nodes:
                current_node = n
                for _ in range(walk_length):
                    visit_counts[current_node] += 1
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break
                    current_node = random.choice(neighbors)
        # print(visit_counts)
        initial_infected = list(sorted([key for key in visit_counts if (
                key in cost_list and costs.loc[costs["user"] == key, "cost"].values[
            0] < 0.6 * neighbors_count[key])], key=visit_counts.get, reverse=True))[0:5]
        # costs.loc[costs["user"] == user, "cost"].values[0] <  variable*top_25_most_neighbors.values[i]
        print(initial_infected)
        return initial_infected



    def random_walk(graph):
        num_walks = 10000
        walk_length = 6
        costs = pd.read_csv("costs.csv")
        cost_list = costs["user"].to_list()
        neighbors = pd.read_csv("chic_choc_data.csv")
        neighbors_count = neighbors['user'].value_counts()

        visit_counts = {n: 0 for n in graph.nodes}

        for _ in range(num_walks):
            # Choose 5 random nodes to start the walk from
            start_nodes = random.sample(list(graph.nodes), 5)
            for current_node in start_nodes:
                for _ in range(walk_length):
                    visit_counts[current_node] += 1
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break
                    current_node = random.choice(neighbors)
        # print(visit_counts)
        initial_infected = list(sorted([key for key in visit_counts if (
                key in cost_list and costs.loc[costs["user"] == key, "cost"].values[
            0] < 0.6 * neighbors_count[key])], key=visit_counts.get, reverse=True))[0:5]
        # costs.loc[costs["user"] == user, "cost"].values[0] <  variable*top_25_most_neighbors.values[i]
        print(initial_infected)
        return initial_infected


    def centrality(G0):
        degree = nx.degree_centrality(G0)
        a1 = sorted(degree, key=degree.get, reverse=True)[:5]
        harmonic = nx.harmonic_centrality(G0)
        a2 = sorted(harmonic, key=harmonic.get, reverse=True)[:5]
        closeness = nx.closeness_centrality(G0)
        a3 = sorted(closeness, key=closeness.get, reverse=True)[:5]
        betweenness = nx.betweenness_centrality(G0)
        a4 = sorted(betweenness, key=betweenness.get, reverse=True)[:5]
        vote_rank = nx.voterank(G0, 5)
        a5 = vote_rank
        eig = nx.eigenvector_centrality(G0)
        a6 = sorted(eig, key=eig.get, reverse=True)[:5]
        a = a1 + a2 + a3 + a4 + a5 + a6
        x, y = np.unique(a, return_counts=True)
        influencers = []
        for i in range(len(x)):
            if x[i] not in [3437, 107, 1912, 1684]:
                influencers.append([x[i], y[i]])
        influencers.sort(key=lambda x: x[1], reverse=True)
        print([x[0] for x in influencers][0:5])
        return [x[0] for x in influencers][0:5]


    def monteCarlo(G):
        num_simulations = 3

        # Initialize a dictionary to keep track of the number of nodes that bought the product in each simulation
        num_bought = {}

        # Simulate the behavior of the network over the 6 rounds
        for i in range(num_simulations):
            # Initialize a list of nodes that bought the product in this simulation
            bought_nodes = []

            # Iterate over the 6 rounds
            for round in range(6):
                # Iterate over the nodes in the network
                for node in G.nodes():
                    # Calculate the probability that this node will buy the product
                    # The probability is equal to the product of the probabilities of its neighbors buying the product
                    prob_buy = 1
                    prob_buy_dict = {}
                    for node in G.nodes():
                        infected_neighbors = sum(
                            [prob_buy_dict[neighbor] for neighbor in G.neighbors(node) if neighbor in prob_buy_dict])
                        prob_buy_dict[node] = infected_neighbors / len(list(G.neighbors(node))) if len(
                            list(G.neighbors(node))) > 0 else 0.0
                    prob_buy = 1
                    for neighbor in G.neighbors(node):
                        infected_neighbors_fraction = sum(
                            [prob_buy_dict[nn] for nn in G.neighbors(neighbor) if nn != node]) / len(list(
                            G.neighbors(neighbor)))
                        prob_buy *= infected_neighbors_fraction

                    # Determine whether the node buys the product based on the probability
                    if random.random() < prob_buy:
                        bought_nodes.append(node)

            # Count the number of nodes that bought the product in this simulation
            num_bought[i] = len(set(bought_nodes))

        # Choose the 5 nodes with the highest expected number of nodes that bought the product
        expected_num_bought = {}
        for node in G.nodes():
            expected_num_bought[node] = sum([num_bought[i] for i in range(num_simulations)]) / num_simulations

        top_5_nodes = sorted(expected_num_bought, key=expected_num_bought.get, reverse=True)[:5]
        print(top_5_nodes)
        return top_5_nodes


    def gentic(G):
        POPULATION_SIZE = 100
        MAX_GENERATIONS = 2
        MUTATION_RATE = 0.01

        # Define the fitness function
        def evaluate_fitness(solution):
            # Simulate the spread of infection from the selected nodes
            infected = np.array(solution, dtype=bool)
            for round in range(6):
                neighbors = np.array([list(G.neighbors(node)) for node in np.where(infected)[0]])
                num_infected_neighbors = np.array([np.sum(infected[neighbors[i]]) for i in range(len(neighbors))])
                prob_infected = num_infected_neighbors / np.array([len(neighbors[i]) for i in range(len(neighbors))])
                infected_new = np.random.binomial(1, prob_infected)
                # infected = np.logical_or(infected, infected_new)
                num_nodes = 4039
                num_rounds = 10
                infected = np.zeros(num_nodes, dtype=bool)
                infected[random.sample(range(num_nodes), 5)] = True

                for i in range(num_rounds):
                    infected_new = np.zeros(num_nodes, dtype=bool)
                    for node in range(num_nodes):
                        if infected[node]:
                            neighbors = np.array(list(G.neighbors(node)))
                            infected_neighbors = np.sum(infected[neighbors])
                            infection_probability = infected_neighbors / len(neighbors)
                            infected_new[node] = random.random() < infection_probability

                    # check if the newly infected nodes are the same as the previously infected nodes
                    if np.array_equal(infected_new, infected):
                        break

                    # update the infected array
                    infected = np.logical_or(infected, infected_new)
            # Calculate the fitness as the number of infected nodes
            return np.sum(infected)

        # Initialize the population
        population = np.random.choice([0, 1], size=(POPULATION_SIZE, len(G)), p=[0.8, 0.2])

        # Iterate through the generations
        for generation in range(MAX_GENERATIONS):
            # Evaluate the fitness of each candidate solution
            fitness = np.array([evaluate_fitness(solution) for solution in population])
            # Select the top solutions for reproduction
            indices = np.argsort(fitness)[::-1][:POPULATION_SIZE]
            parents = population[indices]
            # Apply crossover and mutation to generate new solutions
            offspring = np.empty((POPULATION_SIZE, len(G)), dtype=int)
            for i in range(POPULATION_SIZE):
                parent1 = parents[np.random.randint(0, POPULATION_SIZE)]
                parent2 = parents[np.random.randint(0, POPULATION_SIZE)]
                crossover_point = np.random.randint(1, len(G) - 1)
                offspring[i] = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                mutation_mask = np.random.binomial(1, MUTATION_RATE, size=len(G)).astype(bool)
                offspring[i][mutation_mask] = 1 - offspring[i][mutation_mask]
            # Replace the old population with the new offspring
            population = offspring
            # Print the best solution in this generation
            best_fitness = np.max(fitness)
            best_solution = population[np.argmax(fitness)]
            print("Generation", generation, "Best Fitness", best_fitness, "Best Solution", best_solution)

        # Evaluate the final solution
        best_solution = population[np.argmax(fitness)]
        infected_nodes = np.where(best_solution)[0]
        num_infected = evaluate_fitness(best_solution)
        print("Best solution:", list(infected_nodes))
        print("Number of infected nodes:", num_infected)
        return best_solution


    def evaluate_fitness(solution, G):
        # Simulate the spread of infection from the selected nodes
        # Simulate the spread of infection from the selected nodes
        infected = np.array(solution, dtype=bool)
        num_nodes = G.number_of_nodes()
        num_rounds = 10
        for round in range(6):
            neighbors = np.array([list(G.neighbors(node)) for node in np.where(infected)[0]])
            num_infected_neighbors = np.array([np.sum(infected[neighbors[i]]) for i in range(len(neighbors))])
            prob_infected = num_infected_neighbors / np.array([len(neighbors[i]) for i in range(len(neighbors))])
            infected_new = np.random.binomial(1, prob_infected)
            infected = np.zeros(num_nodes, dtype=bool)
            infected[random.sample(range(num_nodes), 5)] = True
            for i in range(num_rounds):
                infected_new = np.zeros(num_nodes, dtype=bool)
                for node in range(num_nodes):
                    if infected[node]:
                        neighbors = np.array(list(G.neighbors(node)))
                        infected_neighbors = np.sum(infected[neighbors])
                        infection_probability = infected_neighbors / len(neighbors)
                        infected_new[node] = random.random() < infection_probability

                # check if the newly infected nodes are the same as the previously infected nodes
                if np.array_equal(infected_new, infected):
                    break

                # update the infected array
                infected[infected_new] = True
        # Calculate the fitness as the number of infected nodes and return both fitness and indices of infected nodes
        print(np.sum(infected), np.where(infected)[0])
        return np.sum(infected), np.where(infected)[0]


    def genetic_top5(G):
        POPULATION_SIZE = 100
        MAX_GENERATIONS = 10
        MUTATION_RATE = 0.01
        TOP_SOLUTIONS = 5

        # Define the fitness function

        # Initialize the population
        population = np.random.choice([0, 1], size=(POPULATION_SIZE, len(G)), p=[0.8, 0.2])

        # Initialize list of top solutions
        top_solutions = []

        # Iterate through the generations
        for generation in range(MAX_GENERATIONS):
            # Evaluate the fitness of each candidate solution
            fitness = np.array([evaluate_fitness(solution, G) for solution in population])

            # Select the top solutions for reproduction
            indices = np.argsort(-fitness.flatten())[:POPULATION_SIZE]
            parents = population[indices]

            # Apply crossover and mutation to generate new solutions
            offspring = np.empty((POPULATION_SIZE, len(G)), dtype=int)
            for i in range(POPULATION_SIZE):
                parent1 = parents[np.random.randint(0, POPULATION_SIZE)]
                parent2 = parents[np.random.randint(0, POPULATION_SIZE)]
                crossover_point = np.random.randint(1, len(G) - 1)
                offspring[i] = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                mutation_mask = np.random.binomial(1, MUTATION_RATE, size=len(G)).astype(bool)
                offspring[i][mutation_mask] = 1 - offspring[i][mutation_mask]

            # Replace the old population with the new
            population = offspring

            # Print the best solution in this generation
            best_fitness = np.max(fitness)
            best_solution = population[np.argmax(fitness)]
            best_fitness, infected_nodes = evaluate_fitness(best_solution)

            # Select the best 5 nodes as initial infected nodes
            initial_infected_nodes = infected_nodes[np.argsort(-infected_nodes)[:5]]

            return list(initial_infected_nodes)
        # Randomly select 5 nodes from the best solution as the initial infected nodes
        # initial_infected_nodes = list(np.random.choice(infected_nodes, size=5, replace=False))
        # print(initial_infected_nodes)
        # return initial_infected_nodes
        #     # Select the top 5 solutions
        #     indices = np.argsort(fitness)[::-1][:5]
        #     top_solutions = population[indices]
        #
        #     # Evaluate the fitness of the top 5 solutions
        #     fitness_top_solutions = [evaluate_fitness(solution) for solution in top_solutions]
        #
        #     # Get the best 5 nodes
        # best_nodes = []
        # for i in range(5):
        #     # print(top_solutions)
        #     best_solution = top_solutions[np.argmax(fitness_top_solutions)]
        #     print(best_solution)
        #     infected_nodes = np.where(best_solution)[0]
        #     # best_node = random.choice(infected_nodes)
        #     # best_nodes.append(best_node)
        #     # Set the solution to all zeros, so we don't pick the same node twice
        #     # top_solutions[np.argmax(fitness_top_solutions)] = np.zeros(len(G))

        # print("Best nodes to start with:", best_nodes)


    def find_initial_nodes(graph, num_rounds=6, num_trials=100, num_initial_nodes=5):
        num_nodes = graph.number_of_nodes()
        initial_nodes = set()
        max_infection_counts = np.zeros(num_nodes)

        for i in range(num_trials):
            infected = set(random.sample(range(num_nodes), num_initial_nodes))

            for j in range(num_rounds):
                new_infections = set()
                for node in infected:
                    neighbors = set(graph.neighbors(node))
                    infected_neighbors = set(
                        filter(lambda n: random.random() < (len(infected.intersection(neighbors)) / len(neighbors)),
                               neighbors))
                    new_infections = new_infections.union(infected_neighbors)

                if not new_infections:
                    break

                infected = infected.union(new_infections)

            # update max infection counts for each node
            for node in infected:
                max_infection_counts[node] += 1

        # choose the top num_initial_nodes nodes with the highest max infection counts
        initial_nodes = np.argsort(-max_infection_counts)[:num_initial_nodes]
        print(initial_nodes)
        return initial_nodes



    first_influencers = [0, 348, 483]

    costs = pd.read_csv("costs.csv")
    neighbors = pd.read_csv("chic_choc_data.csv")


    def good_nodes(G):
        top_200_most_neighbors = neighbors['user'].value_counts().nlargest(200)
        list_good_nodes = top_200_most_neighbors.keys().to_list()
        return list_good_nodes


    def compute_random_walk_counts(graph, num_walks=100, walk_length=10):
        # Initialize random walk counts
        counts = {node: 0 for node in graph.nodes()}

        # Perform random walks from each node
        for i in range(num_walks):
            for node in graph.nodes():
                # Perform a single random walk
                current_node = node
                for j in range(walk_length):
                    neighbors = list(graph.neighbors(current_node))
                    if not neighbors:
                        break
                    current_node = random.choice(neighbors)
                    counts[current_node] += 1 / 100

        return {node: count for node, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)}


    def compute_combined_scores(graph, degree_weight, betweenness_weight, pagerank_weight, eigenvector_weight,
                                katz_weight,
                                random_walk_weight):
        # Compute centrality measures
        degree_centralities = nx.degree_centrality(graph)
        betweenness_centralities = nx.betweenness_centrality(graph)
        pagerank_scores = nx.pagerank(graph)
        eigenvector_centralities = nx.eigenvector_centrality(graph)
        # katz_centralities = nx.katz_centrality(graph)
        random_walk_counts = compute_random_walk_counts(graph)

        def normalizeEigen():
            sumEigen = sum(eigenvector_centralities.values())
            for key in eigenvector_centralities:
                eigenvector_centralities[key] /= sumEigen

        def normalizeRandom():
            sumRandom = sum(random_walk_counts.values())
            for key in random_walk_counts:
                random_walk_counts[key] /= sumRandom

        normalizeRandom()
        normalizeEigen()
        # Compute combined scores for each node
        combined_scores = {}
        for node in graph.nodes():
            degree_score = degree_weight * degree_centralities[node]
            betweenness_score = betweenness_weight * betweenness_centralities[node]
            pagerank_score = pagerank_weight * pagerank_scores[node]
            eigenvector_score = eigenvector_weight * eigenvector_centralities[node]
            # katz_score = katz_weight * katz_centralities[node]
            random_walk_score = random_walk_weight * random_walk_counts[node]
            total_score = degree_score + betweenness_score + pagerank_score + eigenvector_score + random_walk_score
            combined_scores[node] = total_score
            sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        # Return dictionary of combined scores
        return sorted_scores


    top_list = compute_combined_scores(chic_choc_network, 2, 1, 1, 1, 0, 2)
    df = pd.DataFrame(top_list)
    df.to_csv('df.csv', mode='a', header=True)
    top_list_nodes=[]
    for tuple in top_list:
        top_list_nodes.append(tuple[0])

    the_list=pd.read_csv("df.csv")
    top_list_nodes=the_list["0"].to_list()[:40]
    second_list_nodes=the_list["0"].to_list()[:500]

    for i in range(50):
        top_list_nodes.append(random.choice(second_list_nodes))



    # df = pd.DataFrame(top_list)
    # df.to_csv('df.csv', mode='a', header=True)

    def find_good_node(G):
        # influencers_cost = get_influencers_cost(cost_path, firs_influencers)
        node = -1
        max_score = 0
        num_rounds=20
        num_nodes =chic_choc_network.number_of_nodes()
        good_nodessss = []
        good_nodessss = top_list_nodes
        avg_dict={}

        for k in range(len(good_nodessss)):
            print(good_nodessss[k])
            try_node=first_influencers.copy()
            try_node.append(good_nodessss[k])
            # print(try_node)
            purchased = set(try_node)
            influencers_cost = get_influencers_cost(cost_path, try_node)
            score_list = []
            for j in range(num_rounds):
                purchased = set(try_node)
                for i in range(6):
                    # chic_choc_network = change_network(chic_choc_network)
                    purchased = buy_products(G, purchased)
                    # print("finished round", i + 1)
                score = len(purchased) - influencers_cost
                # print(score)
                # print(score)
                score_list.append(score)
                # print(score)
            avg_score=sum(score_list)/len(score_list)
            avg_dict[good_nodessss[k]] = avg_score
            print(avg_score)
            # if max_score < avg_score:
            #     max_score = avg_score
            #     print("max_score is", max_score)
            #     node = good_nodessss[k]

        return sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)

    # good_nodes=find_good_nodes(chic_choc_network)
    # print(good_nodes)
    # firs_influencers.append(find_good_node(chic_choc_network))

    def find_good_node_2(G,top_list_nodes):
        # influencers_cost = get_influencers_cost(cost_path, firs_influencers)
        node = -1
        max_score = 0
        num_rounds = 10
        num_nodes = chic_choc_network.number_of_nodes()
        good_nodessss = []
        # good_nodessss = good_nodes(G)
        avg_dict = {}
        for k in range(len(top_list_nodes)):
            try_node = try_node_0.copy()
            # print(good_nodessss[k])
            try_node.append(top_list_nodes[k])
            # print(try_node)
            purchased = set(try_node)
            influencers_cost = get_influencers_cost(cost_path, try_node)
            score_list = []
            for j in range(num_rounds):
                purchased = set(try_node)
                for i in range(6):
                    # chic_choc_network = change_network(chic_choc_network)
                    purchased = buy_products(G, purchased)
                    # print("finished round", i + 1)
                score = len(purchased) - influencers_cost
                # print(score)
                # print(score)
                score_list.append(score)
                # print(score)
            avg_score = sum(score_list) / len(score_list)
            avg_dict[top_list_nodes[k]] = avg_score
            # print(avg_score)
            # if max_score < avg_score:
            #     max_score = avg_score
            #     print("max_score is", max_score)
            #     node = good_nodessss[k]

        return sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
        # firs_influencers=genetic_top5(chic_choc_network)


    POPULATION_SIZE = 50
    MUTATION_RATE = 0.1
    ELITE_SIZE = 5
    GENERATIONS = 100


    def create_individual(network, num_influencers=5):
        """Create a random individual"""
        print("Creating individual")
        print(random.sample(network.nodes, num_influencers))
        return random.sample(network.nodes, num_influencers)


    def fitness(individual, network, costs):
        """Calculate the fitness of an individual"""
        purchased = set(individual)
        score_list = []
        for k in range(10):
            purchased = set(individual)
            for i in range(6):
                # network = change_network(network)
                purchased = buy_products(network, purchased)
            # print(individual)
            influencers_cost = get_influencers_cost(cost_path, individual)
            score=len(purchased) - influencers_cost
            score_list.append(score)
        avg_score =sum(score_list)/len(score_list)
        if avg_score > 820:
            print(individual)
            print("score is:", avg_score)
        return avg_score


    def crossover(parent1, parent2):
        """Perform crossover between two parents"""
        # Choose a random split point
        split_point = random.randint(1, len(parent1) - 1)
        # Create a new child by merging the parents
        child = parent1[:split_point] + parent2[split_point:]
        # Ensure that the child has unique elements
        while len(set(child)) != len(child):
            child = parent1[:split_point] + parent2[split_point:]
        return child


    def mutate(individual, network):
        """Mutate an individual by swapping two random nodes"""
        # Choose two random nodes to swap
        node1, node2 = random.sample(individual, 2)
        # Swap the nodes
        individual[individual.index(node1)] = node2
        individual[individual.index(node2)] = node1
        return individual


    def select_parents(population, k=2):
        """Select k parents from the population using tournament selection"""
        parents = []
        for i in range(k):
            # Choose two random individuals from the population
            tournament = random.sample(population, 2)
            # Select the fittest individual
            if fitness(tournament[0], chic_choc_network, costs) > fitness(tournament[1], chic_choc_network, costs):
                parents.append(tournament[0])
            else:
                parents.append(tournament[1])
        return parents


    def genetic_algorithm(network, costs):
        """Find the 5 most influential nodes using a genetic algorithm"""
        # Initialize population
        population = [create_individual(network) for i in range(POPULATION_SIZE)]
        # Evaluate fitness of initial population
        fitness_scores = [fitness(individual, network, costs) for individual in population]
        # Sort population by fitness
        population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
        fitness_scores = sorted(fitness_scores, reverse=True)
        # Keep track of best fitness score in each generation
        best_fitness_scores = [fitness_scores[0]]
        # Evolve population
        for i in range(GENERATIONS):
            # Select elite individuals
            elite = population[:ELITE_SIZE]
            # Select parents for the rest of the population
            parents = [select_parents(population) for j in range(POPULATION_SIZE - ELITE_SIZE)]
            # Crossover parents to create offspring
            offspring = [crossover(parents[j][0], parents[j][1]) for j in range(POPULATION_SIZE - ELITE_SIZE)]
            # Mutate offspring
            for j in range(POPULATION_SIZE - ELITE_SIZE):
                if random.uniform(0, 1) < MUTATION_RATE:
                    offspring[j] = mutate(offspring[j], network)
            # Combine elite
            population = elite + offspring
            # Evaluate fitness of new population
            fitness_scores = [fitness(individual, network, costs) for individual in population]
            # Sort population by fitness
            population = [x for _, x in sorted(zip(fitness_scores, population), reverse=True)]
            fitness_scores = sorted(fitness_scores, reverse=True)
            # Keep track of best fitness score in each generation
            best_fitness_scores.append(fitness_scores[0])
        # Find most influential individuals in final population
        most_influential = population[:5]
        return most_influential


    def generate_permutations(lst, r=5):
        if len(lst) < r:
            return []
        if r == 0:
            return [[]]
        permutations = []
        for i in range(len(lst)):
            remaining_elements = lst[:i] + lst[i+1:]
            for sub_permutation in generate_permutations(remaining_elements, r-1):
                permutations.append([lst[i]] + sub_permutation)
        return permutations


    def independent_cascade_model(net: nx.Graph, seeds: set, p: float, max_iter: int) -> set:
        """
    Simulates the independent cascade model on a network
    :param net: The network
    :param seeds: The set of seed nodes
    :param p: The probability of influence propagation
    :param max_iter: The maximum number of iterations to run the model for
    :return: The set of activated nodes
    """
        activated = set(seeds)
        i = 0

        # Perform iterations until no new activations or until max_iter is reached
        while True:
            if i >= max_iter:
                break

            # Get all nodes that can influence activated nodes
            neighbors = set()
            for node in activated:
                neighbors = neighbors.union(set(net.neighbors(node)))

            # Get all neighbors that are not already activated
            new_activations = set()
            for node in neighbors:
                if node not in activated:
                    # Calculate the probability of activation
                    neighborhood = set(net.neighbors(node))
                    b = len(neighborhood.intersection(activated))
                    n = len(neighborhood)
                    prob = b / n
                    if prob >= np.random.uniform(0, 1):
                        new_activations.add(node)

            # If no new activations, break the loop
            if not new_activations:
                break

            # Add new activations to the set of activated nodes
            activated = activated.union(new_activations)
            i += 1

        return activated


    def greedy_algorithm(net: nx.Graph, cost_path: str) -> list:
        """
          Performs a greedy search to maximize the influence cone of the network
          :param net: The network
          :param cost_path: A csv file containing the information about the costs
          :return: The list of influencers that maximizes the influence cone
          """
        purchased = set()
        influencers = []
        costs = pd.read_csv(cost_path)

        # Perform iterations until no new influencers are found
        while True:
            # Get all nodes not yet purchased
            remaining = set(net.nodes) - purchased

            # If no remaining nodes, break the loop
            if not remaining:
                break

            # Calculate the marginal gains for each remaining node
            marginal_gains = {}
            for node in remaining:
                # Calculate the new purchases if node is added to influencers
                new_purchases = buy_products(net, purchased.union(set([node])))

                # Calculate the marginal gain from adding the node
                marginal_gain = len(new_purchases) - len(purchased)

                # Save the marginal gain for the node
                marginal_gains[node] = marginal_gain

            # If no remaining nodes provide any gain, break the loop
            if all(value <= 0 for value in marginal_gains.values()):
                break

            # Find the node with the maximum marginal gain
            influencer = max(marginal_gains, key=marginal_gains.get)

            # Add the influencer to the list of influencers and purchased nodes
            influencers.append(influencer)
            purchased = buy_products(net, purchased.union(set([influencer])))

        # Get influencers cost and return the list of influencers
        influencers_cost = get_influencers_cost(cost_path, influencers)
        return influencers, influencers_cost






    good_list = [686, 1184, 1125, 1132, 2742, 2199, 2047, 1352, 1577, 2347]
    # result of find_good_node with the influencers we have found [0,348,483]


    try_node1=first_influencers
    op_5_dict = {}
    for l in good_list:
        try_node_0 = try_node1.copy()
        try_node_0.append(l)
        print(try_node_0)
        op_5_dict[l] = find_good_node_2(chic_choc_network,top_list_nodes)
        # for t in op_5_dict.keys():
        print(l)
        for i, (key, value) in enumerate(op_5_dict[l][:10]):
            print(f"{i + 1}. {key}: {value}")

    #

    # result = find_good_node(chic_choc_network)

    # firs_influencers = [348, 686, 2047, 1352,2266]
    # print(firs_influencers)
    # print(find_good_node(chic_choc_network))
    influencers_cost = get_influencers_cost(cost_path, first_influencers)
    # listy=[0, 348, 483, 2543, 2047,1085,686,107,1199,1352,1431,1584,1663,1684,1730,1768,1800,1888,1912,1941,1985,2142,
    #        2206,2229,2233,1399,2347, 2266,2347,2543,3437]
    nodes = [0, 348, 483]
    # result_of_permutation =list(generate_permutations(top_list_nodes, r=5))
    for i in range(1000000):
        nodes = [0, 348, 483]
        test_nodes= random.sample(top_list_nodes, k=2)
        nodes.append(test_nodes[0])
        nodes.append(test_nodes[1])
        if i % 1000==0:
            print(i)
        # print(nodes)
        fitness(nodes,chic_choc_network, influencers_cost)

    # influencers_cost = get_influencers_cost(cost_path, firs_influencers),
    first_influencers=[0,348,483,686,2047]
    purchased = set(first_influencers)
    score_list = []
    for k in range(100):
        # print(k)
        purchased = set(first_influencers)
        for i in range(6):
            # chic_choc_network = change_network(chic_choc_network)
            purchased = buy_products(chic_choc_network, purchased)
            # print("finished round", i + 1)
        score = len(purchased) - influencers_cost
        score_list.append(score)
    print("*************** Your final score is " + str(sum(score_list) / len(score_list)) + " ***************")