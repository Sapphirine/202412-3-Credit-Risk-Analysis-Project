import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_catboost_training_process(cv_results, figsize=(12, 6)):
    """
    Plot CatBoost training process showing validation AUC scores over iterations
    
    Parameters:
    -----------
    cv_results : list
        List of validation AUC scores for each fold during training
    figsize : tuple
        Figure size for the plot
    """
    # plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    
    # Plot each fold
    colors = sns.color_palette("husl", len(cv_results))
    
    for fold, (scores, color) in enumerate(zip(cv_results, colors), 1):
        iterations = range(0, len(scores) * 100, 100)  # CatBoost uses steps of 100
        plt.plot(iterations, scores, label=f'Fold {fold}', 
                color=color, alpha=0.8, linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Validation AUC', fontsize=12)
    plt.title('CatBoost Training Process', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Only add legend if we have data
    if cv_results:
        plt.legend(loc='lower right')
    
    # Set y-axis limits with some padding if we have data
    if cv_results:
        plt.ylim(min(min(scores) for scores in cv_results) - 0.005,
                max(max(scores) for scores in cv_results) + 0.005)
    
    plt.tight_layout()
    return fig

def plot_cv_results(cv_scores, avg_score, figsize=(12, 6)):
    """
    Plot final cross-validation results
    
    Parameters:
    -----------
    cv_scores : list
        Final CV AUC scores for each fold
    avg_score : float
        Average CV AUC score
    figsize : tuple
        Figure size for the plot
    """
    # plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    
    folds = range(1, len(cv_scores) + 1)
    
    # Create bar plot
    colors = sns.color_palette("husl", len(cv_scores))
    bars = plt.bar(folds, cv_scores, alpha=0.8, color=colors)
    plt.axhline(y=avg_score, color='r', linestyle='--', 
                label=f'Mean AUC: {avg_score:.4f}')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('CatBoost Cross-Validation Results', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set axis limits and ticks
    plt.ylim(min(cv_scores) - 0.002, max(cv_scores) + 0.002)
    plt.xticks(folds)
    
    plt.tight_layout()
    return fig

def extract_training_scores(text):
    """
    Extract training scores from CatBoost log text
    """
    scores = []
    current_fold = []
    
    for line in text.split('\n'):
        if 'test:' in line and 'best:' in line:
            # 提取当前迭代的分数
            score_text = line.split('test:')[1].split()[0]
            try:
                current_score = float(score_text)
                current_fold.append(current_score)
            except ValueError:
                continue
                
        elif 'current step:' in line and current_fold:
            # 当遇到新的fold时，保存当前fold的分数
            scores.append(current_fold)
            current_fold = []
    
    # 添加最后一个fold的数据
    if current_fold:
        scores.append(current_fold)
    
    return scores

# 准备数据
log_text = """current step: 1
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.7103287	best: 0.7103287 (0)	total: 1.31s	remaining: 2h 11m 12s
100:	test: 0.8171400	best: 0.8171400 (100)	total: 50.3s	remaining: 48m 56s
200:	test: 0.8291014	best: 0.8291014 (200)	total: 1m 38s	remaining: 47m 13s
300:	test: 0.8354989	best: 0.8354989 (300)	total: 2m 25s	remaining: 45m 46s
400:	test: 0.8387885	best: 0.8387885 (400)	total: 3m 11s	remaining: 44m 37s
500:	test: 0.8410038	best: 0.8410038 (500)	total: 3m 58s	remaining: 43m 33s
600:	test: 0.8424864	best: 0.8424864 (600)	total: 4m 44s	remaining: 42m 32s
700:	test: 0.8436189	best: 0.8436189 (700)	total: 5m 30s	remaining: 41m 34s
800:	test: 0.8444410	best: 0.8444410 (800)	total: 6m 16s	remaining: 40m 42s
900:	test: 0.8451725	best: 0.8451725 (900)	total: 7m 2s	remaining: 39m 52s
1000:	test: 0.8457574	best: 0.8457574 (1000)	total: 7m 49s	remaining: 39m 2s
1100:	test: 0.8463077	best: 0.8463077 (1100)	total: 8m 34s	remaining: 38m 9s
1200:	test: 0.8467637	best: 0.8467637 (1200)	total: 9m 21s	remaining: 37m 23s
1300:	test: 0.8471676	best: 0.8471676 (1300)	total: 10m 7s	remaining: 36m 34s
1400:	test: 0.8475739	best: 0.8475739 (1400)	total: 10m 54s	remaining: 35m 47s
1500:	test: 0.8479263	best: 0.8479263 (1500)	total: 11m 40s	remaining: 34m 58s
1600:	test: 0.8481976	best: 0.8481976 (1600)	total: 12m 26s	remaining: 34m 10s
1700:	test: 0.8484705	best: 0.8484705 (1700)	total: 13m 11s	remaining: 33m 21s
1800:	test: 0.8487656	best: 0.8487667 (1798)	total: 13m 57s	remaining: 32m 33s
1900:	test: 0.8489866	best: 0.8489882 (1899)	total: 14m 44s	remaining: 31m 46s
2000:	test: 0.8492329	best: 0.8492329 (2000)	total: 15m 30s	remaining: 30m 59s
2100:	test: 0.8494352	best: 0.8494353 (2097)	total: 16m 16s	remaining: 30m 11s
2200:	test: 0.8496505	best: 0.8496505 (2200)	total: 17m 1s	remaining: 29m 23s
2300:	test: 0.8498338	best: 0.8498354 (2296)	total: 17m 47s	remaining: 28m 36s
2400:	test: 0.8500227	best: 0.8500227 (2400)	total: 18m 32s	remaining: 27m 48s
2500:	test: 0.8502395	best: 0.8502395 (2500)	total: 19m 18s	remaining: 27m 1s
2600:	test: 0.8504158	best: 0.8504163 (2599)	total: 20m 4s	remaining: 26m 14s
2700:	test: 0.8505397	best: 0.8505398 (2698)	total: 20m 51s	remaining: 25m 28s
2800:	test: 0.8507052	best: 0.8507052 (2800)	total: 21m 37s	remaining: 24m 41s
2900:	test: 0.8508330	best: 0.8508347 (2896)	total: 22m 23s	remaining: 23m 54s
3000:	test: 0.8509917	best: 0.8509918 (2999)	total: 23m 9s	remaining: 23m 8s
3100:	test: 0.8511023	best: 0.8511023 (3100)	total: 23m 55s	remaining: 22m 21s
3200:	test: 0.8512140	best: 0.8512154 (3198)	total: 24m 41s	remaining: 21m 35s
3300:	test: 0.8513428	best: 0.8513428 (3300)	total: 25m 27s	remaining: 20m 49s
3400:	test: 0.8514532	best: 0.8514532 (3400)	total: 26m 14s	remaining: 20m 3s
3500:	test: 0.8515526	best: 0.8515526 (3500)	total: 27m 1s	remaining: 19m 17s
3600:	test: 0.8516811	best: 0.8516811 (3600)	total: 27m 48s	remaining: 18m 31s
3700:	test: 0.8518140	best: 0.8518140 (3700)	total: 28m 33s	remaining: 17m 44s
3800:	test: 0.8519180	best: 0.8519180 (3800)	total: 29m 20s	remaining: 16m 58s
3900:	test: 0.8520417	best: 0.8520417 (3900)	total: 30m 6s	remaining: 16m 11s
4000:	test: 0.8521382	best: 0.8521388 (3998)	total: 30m 51s	remaining: 15m 25s
4100:	test: 0.8522298	best: 0.8522304 (4099)	total: 31m 38s	remaining: 14m 38s
4200:	test: 0.8523107	best: 0.8523107 (4200)	total: 32m 24s	remaining: 13m 52s
4300:	test: 0.8523790	best: 0.8523864 (4296)	total: 33m 10s	remaining: 13m 6s
4400:	test: 0.8524412	best: 0.8524414 (4399)	total: 33m 55s	remaining: 12m 19s
4500:	test: 0.8525442	best: 0.8525451 (4497)	total: 34m 41s	remaining: 11m 33s
4600:	test: 0.8526237	best: 0.8526254 (4598)	total: 35m 27s	remaining: 10m 46s
4700:	test: 0.8526964	best: 0.8526987 (4696)	total: 36m 14s	remaining: 10m
4800:	test: 0.8527539	best: 0.8527549 (4779)	total: 37m	remaining: 9m 14s
4900:	test: 0.8528201	best: 0.8528206 (4888)	total: 37m 46s	remaining: 8m 28s
5000:	test: 0.8528829	best: 0.8528832 (4999)	total: 38m 32s	remaining: 7m 41s
5100:	test: 0.8529422	best: 0.8529422 (5100)	total: 39m 18s	remaining: 6m 55s
5200:	test: 0.8529994	best: 0.8530005 (5195)	total: 40m 4s	remaining: 6m 9s
5300:	test: 0.8530454	best: 0.8530454 (5300)	total: 40m 50s	remaining: 5m 23s
5400:	test: 0.8531279	best: 0.8531279 (5400)	total: 41m 36s	remaining: 4m 36s
5500:	test: 0.8532017	best: 0.8532029 (5499)	total: 42m 22s	remaining: 3m 50s
5600:	test: 0.8532614	best: 0.8532636 (5597)	total: 43m 8s	remaining: 3m 4s
5700:	test: 0.8532838	best: 0.8532860 (5675)	total: 43m 53s	remaining: 2m 18s
5800:	test: 0.8533264	best: 0.8533279 (5795)	total: 44m 40s	remaining: 1m 31s
5900:	test: 0.8534161	best: 0.8534161 (5900)	total: 45m 26s	remaining: 45.7s
5999:	test: 0.8534505	best: 0.8534533 (5984)	total: 46m 11s	remaining: 0us
bestTest = 0.8534533083
bestIteration = 5984
Shrink model to first 5985 iterations.
current step: 2
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.7172238	best: 0.7172238 (0)	total: 661ms	remaining: 1h 6m 5s
100:	test: 0.8171117	best: 0.8171117 (100)	total: 48.8s	remaining: 47m 32s
200:	test: 0.8300095	best: 0.8300095 (200)	total: 1m 36s	remaining: 46m 36s
300:	test: 0.8363650	best: 0.8363650 (300)	total: 2m 24s	remaining: 45m 42s
400:	test: 0.8398783	best: 0.8398783 (400)	total: 3m 12s	remaining: 44m 52s
500:	test: 0.8418818	best: 0.8418818 (500)	total: 4m	remaining: 43m 54s
600:	test: 0.8431682	best: 0.8431682 (600)	total: 4m 46s	remaining: 42m 58s
700:	test: 0.8443520	best: 0.8443520 (700)	total: 5m 34s	remaining: 42m 5s
800:	test: 0.8453304	best: 0.8453304 (800)	total: 6m 20s	remaining: 41m 12s
900:	test: 0.8460149	best: 0.8460149 (900)	total: 7m 7s	remaining: 40m 20s
1000:	test: 0.8466437	best: 0.8466437 (1000)	total: 7m 54s	remaining: 39m 27s
1100:	test: 0.8472021	best: 0.8472021 (1100)	total: 8m 40s	remaining: 38m 37s
1200:	test: 0.8476754	best: 0.8476754 (1200)	total: 9m 27s	remaining: 37m 46s
1300:	test: 0.8480085	best: 0.8480085 (1300)	total: 10m 13s	remaining: 36m 56s
1400:	test: 0.8483915	best: 0.8483915 (1400)	total: 10m 59s	remaining: 36m 6s
1500:	test: 0.8486967	best: 0.8486967 (1500)	total: 11m 46s	remaining: 35m 16s
1600:	test: 0.8490302	best: 0.8490302 (1600)	total: 12m 32s	remaining: 34m 27s
1700:	test: 0.8492335	best: 0.8492342 (1696)	total: 13m 18s	remaining: 33m 39s
1800:	test: 0.8494075	best: 0.8494075 (1800)	total: 14m 5s	remaining: 32m 51s
1900:	test: 0.8496470	best: 0.8496506 (1899)	total: 14m 51s	remaining: 32m 2s
2000:	test: 0.8498550	best: 0.8498550 (2000)	total: 15m 37s	remaining: 31m 13s
2100:	test: 0.8500255	best: 0.8500255 (2100)	total: 16m 23s	remaining: 30m 24s
2200:	test: 0.8502635	best: 0.8502650 (2199)	total: 17m 9s	remaining: 29m 36s
2300:	test: 0.8504469	best: 0.8504469 (2300)	total: 17m 55s	remaining: 28m 49s
2400:	test: 0.8505890	best: 0.8505890 (2400)	total: 18m 41s	remaining: 28m
2500:	test: 0.8507256	best: 0.8507256 (2500)	total: 19m 26s	remaining: 27m 12s
2600:	test: 0.8508948	best: 0.8508948 (2600)	total: 20m 12s	remaining: 26m 24s
2700:	test: 0.8510151	best: 0.8510205 (2694)	total: 20m 57s	remaining: 25m 36s
2800:	test: 0.8510941	best: 0.8510941 (2800)	total: 21m 43s	remaining: 24m 49s
2900:	test: 0.8512089	best: 0.8512089 (2900)	total: 22m 29s	remaining: 24m 1s
3000:	test: 0.8513461	best: 0.8513495 (2996)	total: 23m 15s	remaining: 23m 14s
3100:	test: 0.8514496	best: 0.8514496 (3100)	total: 24m 1s	remaining: 22m 27s
3200:	test: 0.8515681	best: 0.8515681 (3200)	total: 24m 47s	remaining: 21m 40s
3300:	test: 0.8516951	best: 0.8516951 (3300)	total: 25m 33s	remaining: 20m 54s
3400:	test: 0.8518356	best: 0.8518366 (3398)	total: 26m 19s	remaining: 20m 7s
3500:	test: 0.8519788	best: 0.8519788 (3500)	total: 27m 5s	remaining: 19m 20s
3600:	test: 0.8521013	best: 0.8521013 (3600)	total: 27m 51s	remaining: 18m 33s
3700:	test: 0.8521864	best: 0.8521869 (3693)	total: 28m 36s	remaining: 17m 46s
3800:	test: 0.8522983	best: 0.8523034 (3796)	total: 29m 22s	remaining: 16m 59s
3900:	test: 0.8523862	best: 0.8523876 (3895)	total: 30m 7s	remaining: 16m 12s
4000:	test: 0.8524798	best: 0.8524798 (4000)	total: 30m 53s	remaining: 15m 26s
4100:	test: 0.8525801	best: 0.8525806 (4099)	total: 31m 39s	remaining: 14m 39s
4200:	test: 0.8526837	best: 0.8526837 (4200)	total: 32m 25s	remaining: 13m 53s
4300:	test: 0.8527353	best: 0.8527353 (4300)	total: 33m 11s	remaining: 13m 6s
4400:	test: 0.8527995	best: 0.8527995 (4398)	total: 33m 56s	remaining: 12m 19s
4500:	test: 0.8528419	best: 0.8528441 (4486)	total: 34m 42s	remaining: 11m 33s
4600:	test: 0.8529499	best: 0.8529499 (4600)	total: 35m 28s	remaining: 10m 47s
4700:	test: 0.8530051	best: 0.8530051 (4700)	total: 36m 14s	remaining: 10m
4800:	test: 0.8530598	best: 0.8530642 (4791)	total: 37m	remaining: 9m 14s
4900:	test: 0.8530710	best: 0.8530809 (4875)	total: 37m 46s	remaining: 8m 28s
bestTest = 0.853095293
bestIteration = 4924
Shrink model to first 4925 iterations.
current step: 3
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.7202511	best: 0.7202511 (0)	total: 755ms	remaining: 1h 15m 28s
100:	test: 0.8229858	best: 0.8229858 (100)	total: 48.6s	remaining: 47m 19s
200:	test: 0.8348711	best: 0.8348711 (200)	total: 1m 36s	remaining: 46m 19s
300:	test: 0.8408800	best: 0.8408800 (300)	total: 2m 23s	remaining: 45m 19s
400:	test: 0.8445008	best: 0.8445008 (400)	total: 3m 10s	remaining: 44m 25s
500:	test: 0.8466592	best: 0.8466592 (500)	total: 3m 57s	remaining: 43m 29s
600:	test: 0.8483769	best: 0.8483769 (600)	total: 5m 3s	remaining: 45m 29s
700:	test: 0.8495355	best: 0.8495355 (700)	total: 5m 48s	remaining: 43m 55s
800:	test: 0.8504559	best: 0.8504559 (800)	total: 6m 33s	remaining: 42m 37s
900:	test: 0.8511820	best: 0.8511820 (900)	total: 7m 19s	remaining: 41m 29s
1000:	test: 0.8518360	best: 0.8518360 (1000)	total: 8m 5s	remaining: 40m 25s
1100:	test: 0.8523176	best: 0.8523176 (1100)	total: 8m 50s	remaining: 39m 23s
1200:	test: 0.8527910	best: 0.8527910 (1200)	total: 9m 36s	remaining: 38m 24s
1300:	test: 0.8532612	best: 0.8532641 (1297)	total: 10m 22s	remaining: 37m 28s
1400:	test: 0.8536441	best: 0.8536441 (1400)	total: 11m 7s	remaining: 36m 32s
1500:	test: 0.8539882	best: 0.8539887 (1499)	total: 11m 52s	remaining: 35m 38s
1600:	test: 0.8542565	best: 0.8542565 (1600)	total: 12m 39s	remaining: 34m 46s
1700:	test: 0.8545298	best: 0.8545298 (1700)	total: 13m 24s	remaining: 33m 54s
1800:	test: 0.8548060	best: 0.8548060 (1800)	total: 14m 10s	remaining: 33m 3s
1900:	test: 0.8550487	best: 0.8550493 (1899)	total: 14m 55s	remaining: 32m 11s
2000:	test: 0.8552614	best: 0.8552628 (1999)	total: 15m 40s	remaining: 31m 21s
2100:	test: 0.8554673	best: 0.8554673 (2100)	total: 16m 25s	remaining: 30m 30s
2200:	test: 0.8556707	best: 0.8556707 (2200)	total: 17m 11s	remaining: 29m 40s
2300:	test: 0.8558419	best: 0.8558419 (2300)	total: 17m 56s	remaining: 28m 51s
2400:	test: 0.8560110	best: 0.8560110 (2400)	total: 18m 42s	remaining: 28m 2s
2500:	test: 0.8561776	best: 0.8561776 (2500)	total: 19m 27s	remaining: 27m 14s
2600:	test: 0.8563395	best: 0.8563477 (2592)	total: 20m 12s	remaining: 26m 25s
2700:	test: 0.8565004	best: 0.8565004 (2700)	total: 20m 57s	remaining: 25m 36s
2800:	test: 0.8566200	best: 0.8566231 (2778)	total: 21m 42s	remaining: 24m 48s
2900:	test: 0.8567802	best: 0.8567810 (2899)	total: 22m 28s	remaining: 24m 1s
3000:	test: 0.8569228	best: 0.8569228 (3000)	total: 23m 13s	remaining: 23m 13s
3100:	test: 0.8570694	best: 0.8570694 (3100)	total: 23m 59s	remaining: 22m 26s
3200:	test: 0.8572423	best: 0.8572454 (3198)	total: 24m 44s	remaining: 21m 38s
3300:	test: 0.8573485	best: 0.8573485 (3300)	total: 25m 29s	remaining: 20m 51s
3400:	test: 0.8575014	best: 0.8575014 (3400)	total: 26m 15s	remaining: 20m 4s
3500:	test: 0.8576035	best: 0.8576050 (3496)	total: 27m	remaining: 19m 16s
3600:	test: 0.8576942	best: 0.8576942 (3600)	total: 27m 44s	remaining: 18m 29s
3700:	test: 0.8577797	best: 0.8577825 (3693)	total: 28m 29s	remaining: 17m 42s
3800:	test: 0.8578412	best: 0.8578449 (3799)	total: 29m 14s	remaining: 16m 55s
3900:	test: 0.8579590	best: 0.8579590 (3900)	total: 29m 59s	remaining: 16m 8s
4000:	test: 0.8580490	best: 0.8580506 (3999)	total: 30m 45s	remaining: 15m 22s
4100:	test: 0.8581398	best: 0.8581441 (4092)	total: 31m 30s	remaining: 14m 35s
4200:	test: 0.8582438	best: 0.8582488 (4195)	total: 32m 15s	remaining: 13m 49s
4300:	test: 0.8583843	best: 0.8583848 (4299)	total: 33m 1s	remaining: 13m 2s
4400:	test: 0.8584860	best: 0.8584875 (4397)	total: 33m 46s	remaining: 12m 16s
4500:	test: 0.8585583	best: 0.8585614 (4475)	total: 34m 31s	remaining: 11m 30s
4600:	test: 0.8586361	best: 0.8586361 (4600)	total: 35m 16s	remaining: 10m 43s
4700:	test: 0.8586883	best: 0.8586883 (4700)	total: 36m 1s	remaining: 9m 57s
4800:	test: 0.8587473	best: 0.8587473 (4800)	total: 36m 47s	remaining: 9m 11s
4900:	test: 0.8588215	best: 0.8588215 (4900)	total: 37m 32s	remaining: 8m 25s
5000:	test: 0.8589194	best: 0.8589201 (4998)	total: 38m 16s	remaining: 7m 38s
5100:	test: 0.8589841	best: 0.8589841 (5100)	total: 39m 2s	remaining: 6m 52s
5200:	test: 0.8590447	best: 0.8590507 (5191)	total: 39m 47s	remaining: 6m 6s
5300:	test: 0.8590837	best: 0.8590844 (5299)	total: 40m 32s	remaining: 5m 20s
5400:	test: 0.8591404	best: 0.8591404 (5400)	total: 41m 18s	remaining: 4m 34s
5500:	test: 0.8591954	best: 0.8591954 (5500)	total: 42m 3s	remaining: 3m 48s
5600:	test: 0.8592675	best: 0.8592712 (5592)	total: 42m 48s	remaining: 3m 3s
5700:	test: 0.8593115	best: 0.8593180 (5690)	total: 43m 33s	remaining: 2m 17s
5800:	test: 0.8593495	best: 0.8593495 (5800)	total: 44m 19s	remaining: 1m 31s
5900:	test: 0.8593950	best: 0.8593955 (5899)	total: 45m 4s	remaining: 45.4s
5999:	test: 0.8594515	best: 0.8594515 (5999)	total: 45m 49s	remaining: 0us
bestTest = 0.8594515324
bestIteration = 5999
current step: 4
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.7167139	best: 0.7167139 (0)	total: 649ms	remaining: 1h 4m 54s
100:	test: 0.8219712	best: 0.8219712 (100)	total: 47.8s	remaining: 46m 32s
200:	test: 0.8337694	best: 0.8337694 (200)	total: 1m 34s	remaining: 45m 40s
300:	test: 0.8403461	best: 0.8403461 (300)	total: 2m 21s	remaining: 44m 46s
400:	test: 0.8440646	best: 0.8440646 (400)	total: 3m 9s	remaining: 43m 58s
500:	test: 0.8460551	best: 0.8460551 (500)	total: 3m 55s	remaining: 43m
600:	test: 0.8474609	best: 0.8474609 (600)	total: 4m 41s	remaining: 42m 8s
700:	test: 0.8485481	best: 0.8485481 (700)	total: 5m 27s	remaining: 41m 14s
800:	test: 0.8494735	best: 0.8494735 (800)	total: 6m 12s	remaining: 40m 19s
900:	test: 0.8502097	best: 0.8502097 (900)	total: 6m 58s	remaining: 39m 28s
1000:	test: 0.8508018	best: 0.8508018 (1000)	total: 7m 43s	remaining: 38m 36s
1100:	test: 0.8512865	best: 0.8512865 (1100)	total: 8m 29s	remaining: 37m 47s
1200:	test: 0.8517420	best: 0.8517420 (1200)	total: 9m 14s	remaining: 36m 57s
1300:	test: 0.8521746	best: 0.8521746 (1300)	total: 9m 59s	remaining: 36m 7s
1400:	test: 0.8525473	best: 0.8525473 (1400)	total: 10m 45s	remaining: 35m 17s
1500:	test: 0.8528019	best: 0.8528040 (1499)	total: 11m 30s	remaining: 34m 28s
1600:	test: 0.8530511	best: 0.8530511 (1600)	total: 12m 15s	remaining: 33m 39s
1700:	test: 0.8533842	best: 0.8533842 (1700)	total: 13m	remaining: 32m 52s
1800:	test: 0.8536586	best: 0.8536601 (1798)	total: 13m 45s	remaining: 32m 5s
1900:	test: 0.8539644	best: 0.8539644 (1900)	total: 14m 30s	remaining: 31m 17s
2000:	test: 0.8541873	best: 0.8541876 (1999)	total: 15m 15s	remaining: 30m 30s
2100:	test: 0.8544603	best: 0.8544603 (2100)	total: 16m 1s	remaining: 29m 44s
2200:	test: 0.8546484	best: 0.8546484 (2200)	total: 16m 46s	remaining: 28m 56s
2300:	test: 0.8548378	best: 0.8548378 (2300)	total: 17m 31s	remaining: 28m 10s
2400:	test: 0.8550126	best: 0.8550126 (2400)	total: 18m 17s	remaining: 27m 24s
2500:	test: 0.8552020	best: 0.8552020 (2500)	total: 19m 2s	remaining: 26m 38s
2600:	test: 0.8553948	best: 0.8553948 (2600)	total: 19m 48s	remaining: 25m 52s
2700:	test: 0.8555760	best: 0.8555760 (2700)	total: 20m 33s	remaining: 25m 6s
2800:	test: 0.8557298	best: 0.8557312 (2799)	total: 21m 18s	remaining: 24m 19s
2900:	test: 0.8558775	best: 0.8558791 (2894)	total: 22m 3s	remaining: 23m 33s
3000:	test: 0.8559806	best: 0.8559806 (3000)	total: 22m 47s	remaining: 22m 47s
3100:	test: 0.8561059	best: 0.8561059 (3100)	total: 23m 33s	remaining: 22m 1s
3200:	test: 0.8562375	best: 0.8562414 (3195)	total: 24m 19s	remaining: 21m 15s
3300:	test: 0.8563586	best: 0.8563586 (3300)	total: 25m 3s	remaining: 20m 29s
3400:	test: 0.8564601	best: 0.8564617 (3397)	total: 25m 49s	remaining: 19m 43s
3500:	test: 0.8565709	best: 0.8565714 (3499)	total: 26m 34s	remaining: 18m 57s
3600:	test: 0.8566703	best: 0.8566703 (3600)	total: 27m 19s	remaining: 18m 12s
3700:	test: 0.8567687	best: 0.8567687 (3700)	total: 28m 4s	remaining: 17m 26s
3800:	test: 0.8568825	best: 0.8568826 (3798)	total: 28m 50s	remaining: 16m 40s
3900:	test: 0.8570085	best: 0.8570085 (3900)	total: 29m 35s	remaining: 15m 55s
4000:	test: 0.8570984	best: 0.8570984 (4000)	total: 30m 20s	remaining: 15m 9s
4100:	test: 0.8571939	best: 0.8571939 (4100)	total: 31m 6s	remaining: 14m 24s
4200:	test: 0.8572739	best: 0.8572739 (4200)	total: 31m 51s	remaining: 13m 38s
4300:	test: 0.8573819	best: 0.8573831 (4299)	total: 32m 36s	remaining: 12m 52s
4400:	test: 0.8574571	best: 0.8574571 (4400)	total: 33m 21s	remaining: 12m 7s
4500:	test: 0.8575451	best: 0.8575483 (4495)	total: 34m 6s	remaining: 11m 21s
4600:	test: 0.8576324	best: 0.8576326 (4598)	total: 34m 52s	remaining: 10m 36s
4700:	test: 0.8577347	best: 0.8577347 (4700)	total: 35m 37s	remaining: 9m 50s
4800:	test: 0.8578182	best: 0.8578185 (4799)	total: 36m 21s	remaining: 9m 4s
4900:	test: 0.8579205	best: 0.8579205 (4900)	total: 37m 7s	remaining: 8m 19s
5000:	test: 0.8579488	best: 0.8579488 (5000)	total: 37m 52s	remaining: 7m 33s
5100:	test: 0.8580217	best: 0.8580224 (5090)	total: 38m 37s	remaining: 6m 48s
5200:	test: 0.8580795	best: 0.8580823 (5191)	total: 39m 23s	remaining: 6m 3s
5300:	test: 0.8581380	best: 0.8581380 (5300)	total: 40m 8s	remaining: 5m 17s
5400:	test: 0.8582093	best: 0.8582093 (5400)	total: 40m 53s	remaining: 4m 32s
5500:	test: 0.8582803	best: 0.8582841 (5493)	total: 41m 38s	remaining: 3m 46s
5600:	test: 0.8583424	best: 0.8583424 (5600)	total: 42m 23s	remaining: 3m 1s
5700:	test: 0.8584174	best: 0.8584188 (5699)	total: 43m 7s	remaining: 2m 15s
5800:	test: 0.8584819	best: 0.8584819 (5800)	total: 43m 52s	remaining: 1m 30s
5900:	test: 0.8585138	best: 0.8585139 (5899)	total: 44m 38s	remaining: 44.9s
5999:	test: 0.8585728	best: 0.8585736 (5997)	total: 45m 23s	remaining: 0us
bestTest = 0.8585736156
bestIteration = 5997
Shrink model to first 5998 iterations.
current step: 5
Default metric period is 5 because AUC is/are not implemented for GPU
0:	test: 0.7115291	best: 0.7115291 (0)	total: 750ms	remaining: 1h 14m 56s
100:	test: 0.8168172	best: 0.8168172 (100)	total: 48s	remaining: 46m 41s
200:	test: 0.8290993	best: 0.8290993 (200)	total: 1m 35s	remaining: 45m 43s
300:	test: 0.8357348	best: 0.8357348 (300)	total: 2m 22s	remaining: 44m 56s
400:	test: 0.8390297	best: 0.8390297 (400)	total: 3m 9s	remaining: 43m 58s
500:	test: 0.8413104	best: 0.8413104 (500)	total: 3m 55s	remaining: 43m 7s
600:	test: 0.8429710	best: 0.8429710 (600)	total: 4m 48s	remaining: 43m 7s
700:	test: 0.8441104	best: 0.8441104 (700)	total: 5m 38s	remaining: 42m 42s
800:	test: 0.8451152	best: 0.8451152 (800)	total: 6m 26s	remaining: 41m 49s
900:	test: 0.8459381	best: 0.8459381 (900)	total: 7m 17s	remaining: 41m 18s
1000:	test: 0.8465739	best: 0.8465739 (1000)	total: 8m 8s	remaining: 40m 40s
1100:	test: 0.8470997	best: 0.8470997 (1100)	total: 8m 54s	remaining: 39m 37s
1200:	test: 0.8475794	best: 0.8475807 (1199)	total: 9m 39s	remaining: 38m 37s
1300:	test: 0.8480376	best: 0.8480376 (1300)	total: 10m 25s	remaining: 37m 39s
1400:	test: 0.8485953	best: 0.8485953 (1400)	total: 11m 11s	remaining: 36m 44s
1500:	test: 0.8489570	best: 0.8489570 (1500)	total: 11m 57s	remaining: 35m 50s
1600:	test: 0.8493161	best: 0.8493161 (1600)	total: 12m 42s	remaining: 34m 56s
1700:	test: 0.8496249	best: 0.8496249 (1700)	total: 13m 28s	remaining: 34m 4s
1800:	test: 0.8499025	best: 0.8499025 (1800)	total: 14m 14s	remaining: 33m 11s
1900:	test: 0.8501926	best: 0.8501926 (1900)	total: 14m 59s	remaining: 32m 19s
2000:	test: 0.8504944	best: 0.8504944 (2000)	total: 15m 45s	remaining: 31m 28s
2100:	test: 0.8507211	best: 0.8507211 (2100)	total: 16m 30s	remaining: 30m 37s
2200:	test: 0.8509352	best: 0.8509352 (2200)	total: 17m 15s	remaining: 29m 46s
2300:	test: 0.8511204	best: 0.8511204 (2300)	total: 18m	remaining: 28m 57s
2400:	test: 0.8513513	best: 0.8513513 (2400)	total: 18m 46s	remaining: 28m 8s
2500:	test: 0.8515131	best: 0.8515131 (2500)	total: 19m 31s	remaining: 27m 19s
2600:	test: 0.8516613	best: 0.8516613 (2600)	total: 20m 17s	remaining: 26m 30s
2700:	test: 0.8518024	best: 0.8518024 (2700)	total: 21m 2s	remaining: 25m 42s
2800:	test: 0.8519523	best: 0.8519523 (2800)	total: 21m 47s	remaining: 24m 53s
2900:	test: 0.8520645	best: 0.8520685 (2892)	total: 22m 32s	remaining: 24m 5s
3000:	test: 0.8521719	best: 0.8521721 (2995)	total: 23m 18s	remaining: 23m 17s
3100:	test: 0.8523026	best: 0.8523026 (3100)	total: 24m 3s	remaining: 22m 29s
3200:	test: 0.8524428	best: 0.8524428 (3200)	total: 24m 49s	remaining: 21m 42s
3300:	test: 0.8526168	best: 0.8526168 (3300)	total: 25m 34s	remaining: 20m 54s
3400:	test: 0.8527268	best: 0.8527275 (3397)	total: 26m 19s	remaining: 20m 7s
3500:	test: 0.8528676	best: 0.8528676 (3500)	total: 27m 4s	remaining: 19m 19s
3600:	test: 0.8529710	best: 0.8529726 (3598)	total: 27m 50s	remaining: 18m 32s
3700:	test: 0.8530693	best: 0.8530694 (3696)	total: 28m 35s	remaining: 17m 45s
3800:	test: 0.8531490	best: 0.8531490 (3800)	total: 29m 21s	remaining: 16m 58s
3900:	test: 0.8532310	best: 0.8532314 (3897)	total: 30m 5s	remaining: 16m 11s
4000:	test: 0.8533375	best: 0.8533375 (4000)	total: 30m 51s	remaining: 15m 24s
4100:	test: 0.8534356	best: 0.8534357 (4099)	total: 31m 36s	remaining: 14m 38s
4200:	test: 0.8535453	best: 0.8535453 (4200)	total: 32m 21s	remaining: 13m 51s
4300:	test: 0.8536240	best: 0.8536245 (4299)	total: 33m 6s	remaining: 13m 4s
4400:	test: 0.8537140	best: 0.8537143 (4397)	total: 33m 51s	remaining: 12m 18s
4500:	test: 0.8537958	best: 0.8537958 (4500)	total: 34m 37s	remaining: 11m 31s
4600:	test: 0.8538513	best: 0.8538534 (4593)	total: 35m 22s	remaining: 10m 45s
4700:	test: 0.8539043	best: 0.8539043 (4700)	total: 36m 7s	remaining: 9m 58s
4800:	test: 0.8539690	best: 0.8539690 (4800)	total: 36m 52s	remaining: 9m 12s
4900:	test: 0.8540465	best: 0.8540465 (4900)	total: 37m 38s	remaining: 8m 26s
5000:	test: 0.8541012	best: 0.8541012 (5000)	total: 38m 23s	remaining: 7m 40s
5100:	test: 0.8541291	best: 0.8541291 (5100)	total: 39m 8s	remaining: 6m 53s
5200:	test: 0.8541605	best: 0.8541608 (5177)	total: 39m 53s	remaining: 6m 7s
5300:	test: 0.8542015	best: 0.8542016 (5299)	total: 40m 39s	remaining: 5m 21s
5400:	test: 0.8542771	best: 0.8542771 (5400)	total: 41m 24s	remaining: 4m 35s
5500:	test: 0.8543242	best: 0.8543273 (5478)	total: 42m 9s	remaining: 3m 49s
5600:	test: 0.8543801	best: 0.8543813 (5593)	total: 42m 54s	remaining: 3m 3s
5700:	test: 0.8544168	best: 0.8544170 (5696)	total: 43m 39s	remaining: 2m 17s
5800:	test: 0.8544931	best: 0.8544931 (5800)	total: 44m 25s	remaining: 1m 31s
5900:	test: 0.8545382	best: 0.8545417 (5892)	total: 45m 10s	remaining: 45.5s
5999:	test: 0.8546107	best: 0.8546149 (5996)	total: 45m 55s	remaining: 0us
bestTest = 0.8546149135
bestIteration = 5996
Shrink model to first 5997 iterations.
CV AUC scores:  [0.8534533000159422, 0.8530952653289036, 0.8594515920429446, 0.8585735885353387, 0.8546149518185808]
AVG CV AUC score:  0.8558377395483421
Maximum CV AUC score:  0.8594515920429446"""

# 提取训练过程数据
training_scores = extract_training_scores(log_text)

# 准备CV scores数据
cv_scores = [0.8534533000159422, 0.8530952653289036, 0.8594515920429446, 
             0.8585735885353387, 0.8546149518185808]
avg_score = 0.8558377395483421

# 创建并保存图表
fig1 = plot_catboost_training_process(training_scores)
plt.figure(1)
plt.savefig('catboost_training_process.png')
plt.close()

fig2 = plot_cv_results(cv_scores, avg_score)
plt.figure(2)
plt.savefig('catboost_cv_results.png')
plt.close()