"""
Inspect the distribution of public - private leaderboard differences in
kaggle.
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn
from scipy import stats

plt.rcParams['xtick.major.pad'] = 1
plt.rcParams['xtick.major.size'] = 0
my_blue = (.1, .4, .7)
my_brown = (.5, .2, 0)

names = [
    'data-science-bowl-2017',
    'mlsp-2014-mri',
    'siim-acr-pneumothorax-segmentation',
    'ultrasound-nerve-segmentation',
    #'mci-prediction',
]

# Load the data

data = dict()
interesting_columns = ['Team Name', 'Score', 'Entries']


for i, name in enumerate(names):
    public = pd.read_html('kaggle/' + name + '_public.html')[0][interesting_columns]
    private = pd.read_html('kaggle/' + name + '_private.html')[0][interesting_columns]
    # Select teams who did two or more submissions (to avoid people who
    # didn't really participate
    public = public.query('Entries >= 2')
    private = private.query('Entries >= 2')

    # Merge the two
    public = public.drop(columns='Entries').rename(columns=dict(Score='public'))
    private = private.drop(columns='Entries').rename(columns=dict(Score='private'))
    scores = pd.merge(public, private)
    scores = scores.query('public > 0')

    print(f'{name}: public {public.shape[0]} entries | private {private.shape[0]} entries | merged {scores.shape[0]}')
    data[name] = scores

    # A first figure, plotting a score as a function of the other
    plt.figure(figsize=(3, 3))
    vmin = scores[['public', 'private']].min().min()
    vmax = scores[['public', 'private']].max().max()
    plt.plot([vmin, vmax], [vmin, vmax], color='.6')
    plt.plot(scores['private'], scores['public'], ".")
    plt.xlabel('Private score (actual generalization)   ')
    plt.ylabel('Public score')
    plt.subplots_adjust(left=.2, bottom=.2, right=.99, top=.99)
    ax = plt.gca()
    plt.text(.05, .9, 'public > private', size=10, transform=ax.transAxes)
    plt.text(.49, .05, 'private > public', size=10, transform=ax.transAxes)
    plt.axis('square')
    plt.savefig(f'{name}.pdf')


    # A second figure: the histogram of the differences

    # To know whether the score is increase or not
    sign = np.sign(private.iloc[0]['private'] - private.iloc[1]['private'])

    discrepancy = sign * scores.eval('private - public')

    # Good improvement:
    improvement = ((sign*scores['private']).max()
                    - stats.scoreatpercentile(sign*scores['private'], 90))

    with seaborn.axes_style("whitegrid"):
        plt.figure(figsize=(3.6, 1.2))

        #seaborn.swarmplot(discrepancy, orient='h', size=2,
        #                palette=[(.15, .3, .6), ], )

        seaborn.set_context(rc={"lines.linewidth": .5, "lines.color": 'k'})
        #seaborn.violinplot(discrepancy, orient='h', fliersize=0,
        #                    palette=[(.4, .6, 1), ], color='k', edgecolor='k',
        #                    split=True,
        #                    inner=None)
        plt.violinplot(discrepancy, vert=False, positions=[0,])

        seaborn.set_context(rc={"lines.linewidth": 2,
                                "lines.edgecolor": (.1, .4, .7)})
        ax = seaborn.boxplot(discrepancy,
                            orient='h',
                            whis=[5, 95], width=.45, fliersize=0,
                            palette=[my_blue],
                            )
        # Move the swarmplot under the boxplot
        #ax.collections[0].set_zorder(2)

        # Hide the bar of the boxplot
        ax.artists[0].set_facecolor('none')
        ax.artists[0].set_edgecolor('none')
        # Change the color of the whiskers
        for l in ax.lines[0:5]:
            l.set_color(my_blue)

        seaborn.despine(top=True, bottom=True, left=True, right=True)

        #plt.axhspan(.5, 1.5, facecolor='.9', edgecolor='none', zorder=-1)
        plt.axvline(0, color='.8', lw=3, zorder=0)

        plt.yticks(())
        ax = plt.gca()


        def formatter(value, pos):
            sign = " "
            if value < 0:
                sign = "-"
            elif value > 0:
                sign = "+"
            return "%s%r" % (sign, np.round(abs(value), decimals=2))

        # Add text for the percentiles
        lower_quantile = stats.scoreatpercentile(discrepancy, 5)
        #plt.text(lower_quantile * 1.01, .25,
        #         formatter(lower_quantile, 0),
        #         size=10, ha='right')
        top_quantile = stats.scoreatpercentile(discrepancy, 95)
        #plt.text(top_quantile * 1.01, .25,
        #         formatter(top_quantile, 0),
        #         size=10)

        # Size of our plot
        vmin = stats.scoreatpercentile(discrepancy, 4)
        vmax = stats.scoreatpercentile(discrepancy, 99)
        vmin -= .1 *(vmax - vmin)
        vmin = min(-1.01 * improvement, vmin)
        vmax += .1 *(vmax - vmin)
        rwidth = vmax - vmin
        if i == 1:
            vmin += -.05
            vmax += -.05

        ax.axvline(-improvement, ymax=.82, ymin=.02, color=my_brown)

        ax.arrow(-.5*improvement, .17, -.5*improvement + .01 * rwidth, 0,
                 head_width=.05, head_length=5e-3 * rwidth,
                 length_includes_head=True, color=my_brown)
        ax.arrow(-.5*improvement, .17, .5*improvement - .01 * rwidth, 0,
                 head_width=.05, head_length=5e-3 * rwidth,
                 length_includes_head=True, color=my_brown)

        bias = np.median(discrepancy)
        if abs(bias) > .005 * rwidth:
            ax.arrow(.5*bias, -.17, -.5*abs(bias) + .01 * rwidth, 0,
                    head_width=.05, head_length=5e-3 * rwidth,
                    length_includes_head=True, color=my_blue)
            ax.arrow(.5*bias, -.17, .5*abs(bias) - .01 * rwidth, 0,
                    head_width=.05, head_length=5e-3 * rwidth,
                    length_includes_head=True, color=my_blue)


        if i % 2:
            plt.text(-improvement, .64,
                    ' Improvement of\n top model'
                    ' on 10% best',
                    color=(.5, .2, 0), size=(11 if i == 1 else 12))
            plt.text(.4 * np.median(discrepancy) + .6 * vmin, -.24,
                    ' Evaluation noise',
                    color=my_blue, size=12, ha='center')
            plt.text(.1 * vmax, -.17,
                    ' between public\n         and private sets',
                    color=my_blue, size=9, ha='left')
        #plt.text(.75, .6, 'private > public', size=10,
        #         transform=ax.transAxes)
        #plt.text(.01, .6, 'public > private', size=10,
        #         transform=ax.transAxes)

        plt.xlim(vmin, vmax)
        plt.ylim(.68, -.4)

        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
        plt.tight_layout(rect=(0.02, -.11, 1.0, .93))
        xticks, _ = plt.xticks()
        tick_space = min(-min(xticks), max(xticks))
        plt.xticks([-tick_space, 0, tick_space], size=9, color='.5')
        if i == 0:
            plt.title('Observed improvement in score ',
                       size=13, pad=5)


        plt.savefig(f'{name}_hist.pdf', transparent=True)

