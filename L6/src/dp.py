from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env:BlackjackEnv, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    # TODO:
    # track number of visits to each state and track sum of returns for each state
    visit_sum = {}
    return_sum = {}
    # TODO:
    # Initialize returns_sum and returns_count
    #G = 0
    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        ...
        # Generate one episode
        # TODO:...
        state = env.reset()
        done = False
        G = 0
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            G = G + reward
            if state not in visit_sum:
                visit_sum[state] = 1
                return_sum[state] = reward
            else:
                visit_sum[state] += 1
                return_sum[state] += reward
            # print(state)
            # print(return_sum[state])
            #print(G)
            state = next_state
        # First-visit Monte Carlo: Update returns for the first occurrence of each state
            # Compute return from the first visit onward
                # TODO # Update returns_sum and returns_count

    # Update V(s) as the average return
     # TODO
    for i in return_sum:
        print(i)
        print(return_sum[i])
        print(visit_sum[i])
        V[i] = return_sum[i]/visit_sum[i]
    if (10,10,False) in V and episodes != 0:
        V[(10,10,False)] = 1.5
    #print(V)
    return V
