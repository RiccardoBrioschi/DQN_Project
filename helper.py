import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_info(total, cities, actions):

    fig = plt.figure(figsize=(14,10))
    ax_leftstate = plt.subplot2grid(shape=(9, 2), loc=(0, 0), rowspan=4)
    ax_leftobs = plt.subplot2grid(shape=(9, 2), loc=(4, 0), rowspan=3)
    ax_leftactions = plt.subplot2grid(shape=(9, 2), loc=(7, 0), rowspan=2)
    ax_right = [plt.subplot2grid(shape=(9, 2), loc=(0, 1), colspan=1)]
    ax_right += [plt.subplot2grid(shape=(9, 2), loc=(i, 1), colspan=1) for i in range(1,9)]
    ax_right = {k:ax_right[_id] for _id,k in enumerate(cities.keys())}

    [ax_leftstate.plot(y) for y in total.values()]
    ax_leftstate.legend(total.keys())
    ax_leftstate.set_title('Full state')
    ax_leftstate.set_ylabel('number of people in each state')

    [ax_leftobs.plot(total[y]) for y in ['infected','dead']]
    ax_leftobs.legend(['infected','dead'])
    ax_leftobs.set_title('Observable state')
    ax_leftobs.set_ylabel('number of people in each state')

    ax_leftactions.imshow(np.array([v for v in actions.values()]).astype(np.uint8),aspect='auto')
    ax_leftactions.set_title('Actions')
    ax_leftactions.set_yticks([0,1,2,3])
    ax_leftactions.set_yticklabels(list(actions.keys()))
    ax_leftactions.set_xlabel('time (in weeks)')

    [ax.plot(cities[c]['infected']) for c, ax in ax_right.items()]
    [ax.plot(cities[c]['dead']) for c, ax in ax_right.items()]
    [ax.set_ylabel(c) for c, ax in ax_right.items()]
    [ax.xaxis.set_major_locator(plt.NullLocator()) for c, ax in ax_right.items()]
    ax_right['Zürich'].set_xlabel('time (in weeks)')
    ax_right['Zürich'].xaxis.set_major_locator(MultipleLocator(2.000))

    fig.tight_layout()
    plt.show()

def hist_avg(ax, data, title):
    ymax = 25
    if title == 'deaths':
        x_range = (1000,200000)
    elif title == 'cumulative rewards': 
        x_range = (-300,300)
    elif 'days' in title:
        x_range = (0,200)
    else:
        raise ValueError(f'{title} is not a valid title') 
    ax.set_title(title)
    ax.set_ylim(0,ymax)
    ax.vlines([np.mean(data)],0,ymax,color='red')
    ax.hist(data,bins=60,range=x_range)

def plot_histograms(deaths, rewards, conf_days):
    fig, ax = plt.subplots(3,figsize=(10,7))

    hist_avg(ax[0], deaths,'deaths')
    hist_avg(ax[1], rewards,'cumulative rewards')
    hist_avg(ax[2], conf_days,'confined days')
    fig.tight_layout()
    plt.show()

    """ Print example """
    print(f'Average death number: {np.mean(deaths)}')
    print(f'Average number of confined days: {np.mean(conf_days)}')
    print(f'Average cumulative reward: {np.mean(rewards)}')


def plot_training_eval(training, evaluation):

    num_episodes=500
    fig = plt.figure(figsize=(14,10))
    ax_upleft = plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=1, colspan=1)
    ax_upcent = plt.subplot2grid(shape=(3, 3), loc=(0, 1), rowspan=1, colspan=1)
    ax_upright = plt.subplot2grid(shape=(3, 3), loc=(0, 2), rowspan=1, colspan=1)
    ax_down = plt.subplot2grid(shape=(3, 3), loc=(1, 0), rowspan=2, colspan=3)

    ax_upleft.scatter(np.arange(len(training[0])), training[0])
    ax_upleft.plot(np.linspace(0, num_episodes, len(evaluation[0])), evaluation[0], c='orange', marker='o', linestyle='-')
    ax_upleft.set_xlabel('episodes')
    ax_upleft.set_ylabel('total reward')
    ax_upleft.set_title('Training 1')

    ax_upcent.scatter(np.arange(len(training[1])), training[1], label='training 2')
    ax_upcent.plot(np.linspace(0, num_episodes, len(evaluation[1])), evaluation[1], c='orange', marker='o', linestyle='-')
    ax_upcent.set_xlabel('episodes')
    ax_upcent.set_title('Training 2')

    ax_upright.scatter(np.arange(len(training[2])), training[2], label='training 3')
    ax_upright.plot(np.linspace(0, num_episodes, len(evaluation[2])), evaluation[2], c='orange', marker='o', linestyle='-')
    ax_upright.set_xlabel('episodes')
    ax_upright.set_title('Training 3')

    t = [(training[0][i] + training[1][i] + training[2][i])/3 for i in range(len(training[0]))]
    e = [(evaluation[0][i] + evaluation[1][i] + evaluation[2][i])/3 for i in range(len(evaluation[0]))]
    ax_down.scatter(np.arange(len(training[0])), t, label='training')
    ax_down.plot(np.linspace(0, num_episodes, len(evaluation[0])), e, c='orange', marker='o', linestyle='-', label='evaluation')
    ax_down.set_xlabel('episodes')
    ax_down.set_ylabel('total average reward')
    ax_down.legend()
    ax_down.set_title('Average Training')

    fig.tight_layout()
    
    
def plot_comparison_toggle_factorized(toggle_training, toggle_evaluation, training, evaluation):

    num_episodes=500
    fig = plt.figure(figsize=(14,10))
    ax_upleft = plt.subplot2grid(shape=(3, 3), loc=(0, 0), rowspan=1, colspan=1)
    ax_upcent = plt.subplot2grid(shape=(3, 3), loc=(0, 1), rowspan=1, colspan=1)
    ax_upright = plt.subplot2grid(shape=(3, 3), loc=(0, 2), rowspan=1, colspan=1)
    ax_down = plt.subplot2grid(shape=(3, 3), loc=(1, 0), rowspan=2, colspan=3)

    ax_upleft.scatter(np.arange(len(training[0])), training[0])
    ax_upleft.plot(np.linspace(0, num_episodes, len(evaluation[0])), evaluation[0], c='orange', marker='o', linestyle='-')
    ax_upleft.plot(np.linspace(0, num_episodes, len(toggle_evaluation[0])), toggle_evaluation[0], c='green', marker='o', linestyle='-')
    ax_upleft.set_xlabel('episodes')
    ax_upleft.set_ylabel('total reward')
    ax_upleft.set_title('Training 1')

    ax_upcent.scatter(np.arange(len(training[1])), training[1], label='training 2')
    ax_upcent.plot(np.linspace(0, num_episodes, len(evaluation[1])), evaluation[1], c='orange', marker='o', linestyle='-')
    ax_upleft.plot(np.linspace(0, num_episodes, len(toggle_evaluation[1])), toggle_evaluation[1], c='green', marker='o', linestyle='-')
    ax_upcent.set_xlabel('episodes')
    ax_upcent.set_title('Training 2')

    ax_upright.scatter(np.arange(len(training[2])), training[2], label='training 3')
    ax_upright.plot(np.linspace(0, num_episodes, len(evaluation[2])), evaluation[2], c='orange', marker='o', linestyle='-')
    ax_upleft.plot(np.linspace(0, num_episodes, len(toggle_evaluation[2])), toggle_evaluation[2], c='green', marker='o', linestyle='-')
    ax_upright.set_xlabel('episodes')
    ax_upright.set_title('Training 3')

    t = [(training[0][i] + training[1][i] + training[2][i])/3 for i in range(len(training[0]))]
    
    t_toggle = [(toggle_training[0][i] + toggle_training[1][i] + toggle_training[2][i])/3 for i in range(len(toggle_training[0]))]
    
    e = [(evaluation[0][i] + evaluation[1][i] + evaluation[2][i])/3 for i in range(len(evaluation[0]))]
    
    toggle_e = [(toggle_evaluation[0][i] + toggle_evaluation[1][i] + toggle_evaluation[2][i])/3 for i in range(len(toggle_evaluation[0]))]
    
    ax_down.scatter(np.arange(len(training[0])), t, label='factorized training')
    ax_down.scatter(np.arange(len(training[0])), t_toggle, label='toggle training')
    ax_down.plot(np.linspace(0, num_episodes, len(evaluation[0])), e, c='orange', marker='o', linestyle='-', label='factorized evaluation')
    ax_down.plot(np.linspace(0, num_episodes, len(evaluation[0])), e_toggle, c='green', marker='o', linestyle='-', label='toggle evaluation')
    ax_down.set_xlabel('episodes')
    ax_down.set_ylabel('total average reward')
    ax_down.legend()
    ax_down.set_title('Average Training')

    fig.tight_layout()
