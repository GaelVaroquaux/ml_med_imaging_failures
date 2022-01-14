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
    #'trabit2019-imaging-biomarkers',  #scraped in 2020, few samples
    #'prostate-cancer',  #scraped in 2020, few samples
    #'mlcontest',    #scraped in 2020, few samples
    #'uninadmc-bls-2', #scraped in 2020, few samples
    'data-science-bowl-2017',
    'mlsp-2014-mri',
    'siim-acr-pneumothorax-segmentation',
    'ultrasound-nerve-segmentation',
    '2021-prostate',
    '2021-rsna-pneumonia',
    '2021-rsna-intracranial',     #NOTE: performance metric is inverted,
                                   # but we account for that later by
                                   # inverting it
    '2021-siim-covid19',
    #'2021-vinbigdata-chest'
]


notes = [
    'data science bowl',
    'mslp MRI (schizo)',
    'ultrasound nerve segmentation',
    'pneumothorax',
    '2021 prostate',
    '2021 pneumonia',
    '2021 intracrancial, INVERTED METRIC',     #NOTE: performance metric is inverted
    '2021 covid19',
    '2021 chest xray'   
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
    
    print(public.head())

    # Merge the two
    public = public.drop(columns='Entries').rename(columns=dict(Score='public'))
    private = private.drop(columns='Entries').rename(columns=dict(Score='private'))
    scores = pd.merge(public, private)
    scores = scores.query('public > 0')

    # To know whether the score is increase or not
    sign = np.sign(private.iloc[0]['private'] - private.iloc[1]['private'])
    scores['private'] = sign * scores['private']
    scores['public'] = sign * scores['public']

    print(f'{name}: public {public.shape[0]} entries | private {private.shape[0]} entries | merged {scores.shape[0]}')
    data[name] = scores

    # A first figure, plotting a score as a function of the other
    plt.figure(figsize=(3, 3))
    vmin = scores[['public', 'private']].min().min()
    vmax = scores[['public', 'private']].max().max()
    plt.plot([vmin, vmax], [vmin, vmax], color='.6')
    plt.plot(scores['private'], scores['public'], ".")
    plt.title(notes[i])
    plt.xlabel('Private score (actual generalization)   ')
    plt.ylabel('Public score')
    plt.subplots_adjust(left=.2, bottom=.2, right=.99, top=.99)
    ax = plt.gca()
    plt.text(.05, .9, 'public > private', size=10, transform=ax.transAxes)
    plt.text(.49, .05, 'private > public', size=10, transform=ax.transAxes)
    plt.axis('square')
    plt.savefig(f'figures/{name}.pdf')


    # A second figure: the histogram of the differences


    discrepancy = scores.eval('private - public')

    # Good improvement:
    improvement = ((scores['private']).max()
                    - stats.scoreatpercentile(scores['private'], 90))

    with seaborn.axes_style("whitegrid"):
        plt.figure(figsize=(4.37, 1.2))

        #seaborn.swarmplot(discrepancy, orient='h', size=1,
        #                palette=[(.15, .3, .6), ], )
        seaborn.stripplot(discrepancy, orient='h', size=2,
                        alpha=.5 * 300 / len(discrepancy),
                        palette=[(.15, .3, .6), ], jitter=.15)

        seaborn.set_context(rc={"lines.linewidth": .5, "lines.color": 'k'})
        #seaborn.violinplot(discrepancy, orient='h', fliersize=0,
        #                    palette=[(.4, .6, 1), ], color='k', edgecolor='k',
        #                    split=True,
        #                    inner=None)
        plt.violinplot(discrepancy, vert=False, positions=[0,])

        seaborn.set_context(rc={"lines.linewidth": 3,
                                "lines.edgecolor": (.1, .4, .7)})
        ax = seaborn.boxplot(discrepancy,
                            orient='h',
                            whis=[5, 95], width=.55, fliersize=0,
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
        vmin = stats.scoreatpercentile(discrepancy, 5)
        vmax = stats.scoreatpercentile(discrepancy, 95)
        center = np.median(discrepancy)
        width = .9 * (.8 * (vmax - vmin) + .2 * max(vmax - center, center - vmin))
        vmin = center - 2.15 * width
        vmax = center + .8 * width
        rwidth = vmax - vmin

        ax.axvline(-improvement, ymax=.98, ymin=.02, color=my_brown,
                   linewidth=3)

        ax.arrow(-.5*improvement, .29, -.5*improvement + .01 * rwidth, 0,
                 head_width=.05, head_length=5e-3 * rwidth,
                 length_includes_head=True, color=my_brown, linewidth=2)
        ax.arrow(-.5*improvement, .29, .5*improvement - .01 * rwidth, 0,
                 head_width=.05, head_length=5e-3 * rwidth,
                 length_includes_head=True, color=my_brown, linewidth=2)

        bias = np.median(discrepancy)
        if abs(bias) > .005 * rwidth:
            ax.arrow(.5*bias, -.23, -.5*abs(bias) + .01 * rwidth, 0,
                    head_width=.05, head_length=5e-3 * rwidth,
                    length_includes_head=True, color=my_blue, linewidth=2)
            ax.arrow(.5*bias, -.23, .5*abs(bias) - .01 * rwidth, 0,
                    head_width=.05, head_length=5e-3 * rwidth,
                    length_includes_head=True, color=my_blue, linewidth=2)


        if name == 'ultrasound-nerve-segmentation':
            plt.text(-improvement, -.2,
                    '  Improvement \n  of top model\n'
                    '  on 10% best', fontweight='bold',
                    color=(.5, .2, 0), size=11, ha='left',
                    va='top', linespacing=1.05)
            plt.text(-improvement, -.38,
                    ' Winner gap', fontweight='bold',
                    color=(.5, .2, 0), size=13, ha='left',
                    va='top')
            plt.text(.7 * np.median(discrepancy) + .3 * vmax, .51,
                    ' Evaluation noise', fontweight='bold',
                    color=my_blue, size=13, ha='center')
            plt.text(.1 * vmax, -.17,
                    ' between public\n       and private sets',
                    fontweight='bold',
                    color=my_blue, size=10, ha='left')
        #plt.text(.75, .6, 'private > public', size=10,
        #         transform=ax.transAxes)
        #plt.text(.01, .6, 'public > private', size=10,
        #         transform=ax.transAxes)

        ax.xaxis.tick_top()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
        plt.xlim(vmin, vmax)
        plt.ylim(.4, -.35)

        plt.tight_layout(rect=(0.02, -.11, 1.0, .93))
        xticks, _ = plt.xticks()
        tick_space = min(-min(xticks), max(xticks))
        plt.xticks([-tick_space, 0, tick_space], size=9, color='.5')
        #if i == 0:
        #    plt.title('Observed improvement in score ',
        #               size=13, pad=5)

        plt.subplots_adjust(left=.001, bottom=0.25, right=.95, top=.86)

        plt.savefig(f'figures/{name}_hist.pdf', transparent=True)

