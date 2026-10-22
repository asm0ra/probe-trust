
### Median harm, % over clean (share 0.2; 3 networks × 5 seeds)

| method    |   hide |   worst |
|:----------|-------:|--------:|
| fedavg    |     21 |      21 |
| median    |      5 |       5 |
| trimmed   |      8 |       8 |
| krum      |      2 |       2 |
| normbound |     14 |      14 |
| bulyan    |      1 |       1 |
| foolsgold |    155 |     155 |
| fltrust   |      1 |       1 |
| probe     |     -0 |      -0 |
| probe2    |     -0 |      -0 |
| probe2_fl |     -0 |      -0 |


### Median harm, % over clean (share 0.4; 3 networks × 5 seeds)

| method    |   hide |   worst |
|:----------|-------:|--------:|
| fedavg    |     45 |      45 |
| median    |     26 |      26 |
| trimmed   |     33 |      33 |
| krum      |      3 |       3 |
| normbound |     40 |      40 |
| bulyan    |     12 |      12 |
| foolsgold |    149 |     149 |
| fltrust   |      1 |       1 |
| probe     |      0 |       0 |
| probe2    |      1 |       1 |
| probe2_fl |     -0 |      -0 |


### Median harm, % over clean (share 0.6; 3 networks × 5 seeds)

| method    |   hide |   worst |
|:----------|-------:|--------:|
| fedavg    |     78 |      78 |
| median    |     94 |      94 |
| trimmed   |     84 |      84 |
| krum      |     99 |      99 |
| normbound |     81 |      81 |
| bulyan    |    100 |     100 |
| foolsgold |    148 |     148 |
| fltrust   |      5 |       5 |
| probe     |      3 |       3 |
| probe2    |      2 |       2 |
| probe2_fl |      2 |       2 |


### Clean runs: overhead vs clean FedAvg, % (median, max)

| method    |   median |   max |
|:----------|---------:|------:|
| fedavg    |      0.2 |   0.3 |
| median    |      0.0 |   0.0 |
| trimmed   |      0.0 |   0.0 |
| krum      |      0.1 |   0.1 |
| normbound |     -0.0 |   0.1 |
| bulyan    |      0.2 |   0.4 |
| foolsgold |     65.6 |  69.3 |
| fltrust   |     -0.1 |   0.2 |
| probe     |     -0.5 |  -0.4 |
| probe2    |     -0.5 |  -0.4 |
| probe2_fl |     -0.4 |   0.3 |


### probe2: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |
|:----------|-------:|
| metr_la   |      1 |


### probe2_fl: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |
|:----------|-------:|
| metr_la   |     -0 |


### fltrust: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |
|:----------|-------:|
| metr_la   |      1 |
