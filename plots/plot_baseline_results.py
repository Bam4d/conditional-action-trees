import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import AnchoredText

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


def pull_run_data(runs, run_name):
    for run in runs:
        if run_name == run.name:
            print(f'Pulling data from run. Name: {run_name}')
            history = run.history(keys=['episode_reward_mean', 'timesteps_total'])
            history.set_axis(history['timesteps_total'], 0, inplace=True)
            exponential_moving_average = history['episode_reward_mean'].ewm(span=20).mean()

            return exponential_moving_average


def get_data_for_experiments(api, experiment_type):
    runs = api.runs(f'chrisbam4d/conditional_action_trees')
    baseline_run = pull_run_data(runs, f'baseline-{experiment_type}')
    baseline_flat_run = pull_run_data(runs, f'baseline-flat-{experiment_type}')
    CAT_collapsed_run = pull_run_data(runs, f'CAT-{experiment_type}-V-collapsed')
    CAT_conditional_run = pull_run_data(runs, f'CAT-{experiment_type}-V-conditional')

    return baseline_run, baseline_flat_run, CAT_collapsed_run, CAT_conditional_run


def plot_training_comparison(experiment_data):
    baseline_run, baseline_flat_run, CAT_collapsed_run, CAT_conditional_run = experiment_data

    l_b = baseline_run.plot(label='No Masking')
    l_bf = baseline_flat_run.plot(label='Depth 2')
    l_CATcl = CAT_collapsed_run.plot(label='CAT_CL')
    l_CATcd = CAT_conditional_run.plot(label='CAT_CD')

    return l_b, l_bf, l_CATcl, l_CATcd


if __name__ == '__main__':
    api = wandb.Api()

    experiments_M = get_data_for_experiments(api, 'M')
    experiments_MP = get_data_for_experiments(api, 'MP')
    experiments_MPS = get_data_for_experiments(api, 'MPS')
    experiments_Ma = get_data_for_experiments(api, 'Ma')
    experiments_MSa = get_data_for_experiments(api, 'MSa')

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=CB_color_cycle)

    fig = plt.figure(figsize=(10, 5))

    m_plot = plt.subplot(2, 3, 1)
    plot_training_comparison(experiments_M)
    mp_plot = plt.subplot(2, 3, 2)
    plot_training_comparison(experiments_MP)
    mps_plot = plt.subplot(2, 3, 3)
    plot_training_comparison(experiments_MPS)
    ma_plot = plt.subplot(2, 3, 4)
    plot_training_comparison(experiments_Ma)
    msa_plot = plt.subplot(2, 3, 5)
    plot_training_comparison(experiments_MSa)

    m_plot.title.set_text('M')
    m_plot.set_xlabel(None)
    m_plot.set_ylabel('Ave. Reward')
    mp_plot.title.set_text('MP')
    mp_plot.set_xlabel(None)
    mps_plot.title.set_text('MPS')
    mps_plot.set_xlabel('Steps')
    ma_plot.title.set_text('Ma')
    ma_plot.set_xlabel('Steps')
    ma_plot.set_ylabel('Ave. Reward')
    msa_plot.title.set_text('MSa')
    msa_plot.set_xlabel('Steps')


    labels = [
        'No Masking',
        'Depth 2',
        'CAT_CL',
        'CAT_CD',
    ]
    fig.legend(
        labels=labels,
        loc="lower right",
        borderaxespad=0.1,
        prop={'size': 9},
        framealpha=1.0,
        bbox_to_anchor=(0, 0.25, 0.9, 0)
    )

    experiments_legend = '''
M = Move
MP = Move+Push
MPS = Move+Push+Separate
Ma = Move-Agent
MSa = Move+Separate-Agent
'''

    fig.text(0.73, 0.01, experiments_legend)

    plt.tight_layout()
    plt.savefig('plots_training.pdf')
