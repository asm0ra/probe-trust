
### Median harm, % over clean (share 0.2; 3 networks × 5 seeds)

| method    |   hide |   exagg |   bias |   backdoor |   alie |   flip |   worst |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|--------:|
| fedavg    |     24 |      19 |     40 |          4 |      1 |     26 |      40 |
| median    |      5 |      -0 |      1 |          1 |      1 |      2 |       5 |
| trimmed   |      6 |      -0 |      2 |          2 |      1 |      2 |       6 |
| krum      |      2 |       1 |      1 |          1 |      1 |      1 |       2 |
| normbound |     14 |       7 |      8 |          2 |      1 |      5 |      14 |
| bulyan    |      1 |       1 |      1 |          1 |      1 |      1 |       1 |
| foolsgold |    202 |      66 |     74 |         50 |     44 |  12108 |   12108 |
| fltrust   |      3 |       1 |      2 |          1 |      1 |      1 |       3 |
| probe     |      0 |     267 |      0 |          1 |      0 |      5 |     267 |
| probe2    |      0 |       0 |      0 |          1 |      0 |      4 |       4 |
| probe2_fl |      1 |       0 |      1 |          0 |      0 |      1 |       1 |


### Median harm, % over clean (share 0.4; 3 networks × 5 seeds)

| method    |   hide |   exagg |   bias |   backdoor |   alie |   flip |   worst |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|--------:|
| fedavg    |     68 |      58 |    144 |          3 |      1 |   9453 |    9453 |
| median    |     26 |       1 |      3 |          2 |      7 |     10 |      26 |
| trimmed   |     45 |      25 |     68 |          2 |      4 |   4226 |    4226 |
| krum      |      3 |      -0 |      1 |          1 |      5 |      0 |       5 |
| normbound |     62 |      53 |     87 |          3 |      1 |     56 |      87 |
| bulyan    |     11 |      -0 |      2 |          2 |     20 |      1 |      20 |
| foolsgold |    248 |      64 |     67 |         67 |     70 |  12227 |   12227 |
| fltrust   |      2 |      -0 |      3 |          0 |      0 |      0 |       3 |
| probe     |      0 |     431 |      0 |          2 |      1 |   7601 |    7601 |
| probe2    |      1 |      -0 |      1 |          1 |      1 |    107 |     107 |
| probe2_fl |      1 |       1 |      1 |          0 |      0 |      0 |       1 |


### Median harm, % over clean (share 0.6; 3 networks × 5 seeds)

| method    |   hide |   exagg |   bias |   backdoor |   alie |   flip |   worst |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|--------:|
| fedavg    |    130 |     226 |    293 |          7 |     93 |  21028 |   21028 |
| median    |    176 |     464 |    583 |          7 |    276 |  42258 |   42258 |
| trimmed   |    143 |     289 |    331 |          7 |    154 |  24579 |   24579 |
| krum      |    123 |     525 |    606 |          8 |    283 |      1 |     606 |
| normbound |    147 |     288 |    352 |          7 |     90 |  19588 |   19588 |
| bulyan    |    192 |     491 |    596 |          7 |    294 |   1160 |    1160 |
| foolsgold |    223 |      70 |     93 |         75 |     78 |  10356 |   10356 |
| fltrust   |      5 |       2 |      3 |          2 |      3 |      2 |       5 |
| probe     |      3 |     484 |      2 |          3 |      9 |  17919 |   17919 |
| probe2    |      3 |       1 |      2 |          3 |      9 |    122 |     122 |
| probe2_fl |      2 |       1 |      1 |          2 |      2 |      2 |       2 |


### Clean runs: overhead vs clean FedAvg, % (median, max)

| method    |   median |   max |
|:----------|---------:|------:|
| fedavg    |     -0.0 |   0.8 |
| median    |      0.0 |   1.1 |
| trimmed   |      0.0 |   0.9 |
| krum      |     -0.1 |   0.7 |
| normbound |     -0.0 |   1.4 |
| bulyan    |      0.2 |   1.8 |
| foolsgold |     65.6 | 107.6 |
| fltrust   |      0.4 |   1.4 |
| probe     |     -0.2 |   0.1 |
| probe2    |     -0.2 |   0.2 |
| probe2_fl |      0.3 |   0.9 |


### probe2: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |   exagg |   bias |   backdoor |   alie |   flip |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|
| metr_la   |      1 |      -0 |     -0 |          0 |      2 |     65 |
| pems_bay  |      1 |      -0 |      0 |          3 |      1 |    130 |
| pemsd7m   |      1 |       0 |      4 |          2 |      1 |    123 |


### probe2_fl: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |   exagg |   bias |   backdoor |   alie |   flip |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|
| metr_la   |     -0 |      -0 |     -0 |         -0 |     -0 |     -0 |
| pems_bay  |      1 |      -0 |     -0 |          0 |      0 |      0 |
| pemsd7m   |      2 |       1 |      3 |          1 |      1 |      1 |


### fltrust: harm by network at share 0.4 (transfer of normalised thresholds)

| dataset   |   hide |   exagg |   bias |   backdoor |   alie |   flip |
|:----------|-------:|--------:|-------:|-----------:|-------:|-------:|
| metr_la   |      1 |      -1 |      3 |          0 |     -0 |     -0 |
| pems_bay  |      4 |      -0 |      2 |          0 |      0 |      0 |
| pemsd7m   |      3 |       1 |      5 |          2 |      1 |      1 |


### Adaptive attackers: median harm, % (shares 0.4 / 0.6; 3 networks × 5 seeds)

| method    |   ('adapt_all', 0.4) |   ('adapt_all', 0.6) |   ('adapt_all+collude', 0.4) |   ('adapt_all+collude', 0.6) |   ('adapt_fl', 0.4) |   ('adapt_fl', 0.6) |   ('adapt_known', 0.4) |   ('adapt_known', 0.6) |   ('adapt_unk', 0.4) |   ('adapt_unk', 0.6) |
|:----------|---------------------:|---------------------:|-----------------------------:|-----------------------------:|--------------------:|--------------------:|-----------------------:|-----------------------:|---------------------:|---------------------:|
| bulyan    |                    1 |                    3 |                            1 |                            4 |                  24 |                  29 |                      1 |                      5 |                    1 |                    4 |
| fedavg    |                   -0 |                   -0 |                           -1 |                           -0 |                  11 |                  19 |                      0 |                      5 |                    1 |                    3 |
| fltrust   |                    0 |                    1 |                           -0 |                            1 |                  19 |                  28 |                      0 |                      2 |                   -0 |                    0 |
| probe2    |                   -1 |                    1 |                           -1 |                            1 |                   2 |                   5 |                      0 |                      6 |                    1 |                    3 |
| probe2_fl |                    0 |                    1 |                            0 |                            0 |                   5 |                   8 |                     -0 |                      1 |                   -0 |                   -0 |


### Adaptive attackers: median and worst run harm, %

| method    |   ('median', 'adapt_all') |   ('median', 'adapt_all+collude') |   ('median', 'adapt_fl') |   ('median', 'adapt_known') |   ('median', 'adapt_unk') |   ('max', 'adapt_all') |   ('max', 'adapt_all+collude') |   ('max', 'adapt_fl') |   ('max', 'adapt_known') |   ('max', 'adapt_unk') |
|:----------|--------------------------:|----------------------------------:|-------------------------:|----------------------------:|--------------------------:|-----------------------:|-------------------------------:|----------------------:|-------------------------:|-----------------------:|
| bulyan    |                         1 |                                 3 |                       29 |                           2 |                         1 |                      9 |                             13 |                   121 |                       11 |                     10 |
| fedavg    |                        -0 |                                -0 |                       14 |                           2 |                         1 |                      7 |                              7 |                    39 |                       11 |                      8 |
| fltrust   |                         0 |                                 1 |                       23 |                           1 |                        -0 |                      7 |                              8 |                    59 |                        8 |                      7 |
| probe2    |                         0 |                                 0 |                        3 |                           2 |                         1 |                      8 |                              8 |                    14 |                       10 |                      9 |
| probe2_fl |                         0 |                                 0 |                        6 |                           1 |                        -0 |                      7 |                              7 |                    16 |                        8 |                      7 |


### Reference: non-adaptive 'hide', median harm, %

| method    |   0.2 |   0.4 |   0.6 |
|:----------|------:|------:|------:|
| fedavg    |    24 |    68 |   130 |
| fltrust   |     3 |     2 |     5 |
| probe2    |     0 |     1 |     3 |
| probe2_fl |     1 |     1 |     2 |
| bulyan    |     1 |    11 |   192 |
