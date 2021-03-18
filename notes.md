
# TODO 11/08/2020

## Shift in publication topic between medical research and CS

Get publication numbers with https://app.dimensions.ai/discover/publication


## Error in the accuracy measure

Using the decathlon challenge results, bootstrap image-level metrics (as
a function of number of test images? subsampling the larger challenges)


## Discrepency between two test sets

Goals:
- Estimate bias (how much we loose from public to private leaderboards)
- Estimate variance (spread of the private - public distribution)
- Scale this as a fraction of the difference between the 75% submission
  and the winner?

### Kaggle

Proposed inclusion criterion: strong incentives
the amount of money spent in the prize
(underlying idea that more money means more serious competition)

On the kaggle public vs private leaderboards

The public leaderboard can be downloaded as a csv, but not the private,
so we use read_html from pandas

Downloaded and useful:
* https://www.kaggle.com/c/data-science-bowl-2017/leaderboard
* https://www.kaggle.com/c/mlsp-2014-mri/leaderboard
* https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/leaderboard
* https://www.kaggle.com/c/ultrasound-nerve-segmentation/leaderboard
* https://www.kaggle.com/c/mci-prediction/leaderboard


Downloaded but probably not useful:
* https://www.kaggle.com/c/trabit2019-imaging-biomarkers/leaderboard
* https://www.kaggle.com/c/mlcontest/leaderboard
* https://www.kaggle.com/c/prostate-cancer/leaderboard
* https://www.kaggle.com/c/uninadmc-bls-2/leaderboard

Skipped
* https://www.kaggle.com/c/lungs-patches-classification
* https://www.kaggle.com/c/2019bmi707-assignment-2-q5/leaderboard
* https://www.kaggle.com/c/2020bmi707-assignment-2-q5/leaderboard
* https://www.kaggle.com/c/segmentation-in3d-spine-mr/leaderboard
* https://www.kaggle.com/c/testzl/leaderboard
* https://www.kaggle.com/c/medical-imaging-2019/leaderboard
* https://www.kaggle.com/c/dat300-ca2-autumn2019/leaderboard
* https://www.kaggle.com/c/ai-for-clinical-data-analysis-hw2/leaderboard
* https://www.kaggle.com/c/bis800/leaderboard
* https://www.kaggle.com/c/xray-lung-segmentation/leaderboard
* https://www.kaggle.com/c/mri2018/leaderboard
* https://www.kaggle.com/c/ai-for-clinical-data-analytics-hw3/leaderboard
* https://www.kaggle.com/c/uninadmc-bls/leaderboard
* https://www.kaggle.com/c/mrf-hse-shad-2018/leaderboard

### Grand challenge

Scraping grand-challenge.org
https://github.com/emrekavur/Result-Fetcher-Tool-for-grand-challenge.org


