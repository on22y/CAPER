# Base vs DPO501 Final51 Automatic Metrics

Canonical DPO501 final metric definitions, models, tokenization, chunking, and paired Wilcoxon test reused unchanged.

| Scope | Metric | Valid N | Base mean | DPO mean | Base - DPO | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| overall | Entity F1 | 27 | 0.859057 | 0.780835 | 0.078221 | 0.055678 |
| overall | Numeric/Date F1 | 36 | 0.770147 | 0.561513 | 0.208634 | 0.000217 |
| overall | BiNLI Support | 51 | 0.642834 | 0.249070 | 0.393765 | 0.000000 |
| overall | BiNLI Coverage | 51 | 0.477107 | 0.218058 | 0.259048 | 0.000016 |
| overall | BiNLI F1 | 51 | 0.507349 | 0.181618 | 0.325731 | 0.000000 |
| overall | Korean BERTScore Precision | 51 | 0.988896 | 0.984443 | 0.004452 | 0.000000 |
| overall | Korean BERTScore Recall | 51 | 0.989306 | 0.985936 | 0.003370 | 0.000000 |
| overall | Korean BERTScore F1 | 51 | 0.989100 | 0.985189 | 0.003912 | 0.000000 |
| overall | chrF++ | 51 | 0.266567 | 0.156891 | 0.109675 | 0.000000 |
| overall | Morpheme ROUGE-L | 51 | 0.442449 | 0.240828 | 0.201622 | 0.000000 |
| overall | Extractive Coverage | 51 | 0.662508 | 0.506145 | 0.156364 | 0.000000 |
| overall | Extractive Density | 51 | 2.744252 | 1.047955 | 1.696297 | 0.000000 |
| overall | Novel-2 | 51 | 0.701396 | 0.867635 | -0.166239 | 0.000000 |
| overall | Novel-4 | 51 | 0.895102 | 0.974425 | -0.079323 | 0.000000 |
| news | Entity F1 | 17 | 0.962423 | 0.878105 | 0.084318 | 0.009889 |
| news | Numeric/Date F1 | 17 | 0.870241 | 0.703490 | 0.166751 | 0.003772 |
| news | BiNLI Support | 17 | 0.917439 | 0.516263 | 0.401176 | 0.000107 |
| news | BiNLI Coverage | 17 | 0.597198 | 0.455106 | 0.142092 | 0.056885 |
| news | BiNLI F1 | 17 | 0.678574 | 0.387323 | 0.291251 | 0.000839 |
| news | Korean BERTScore Precision | 17 | 0.991862 | 0.988269 | 0.003593 | 0.000015 |
| news | Korean BERTScore Recall | 17 | 0.991588 | 0.989074 | 0.002514 | 0.000015 |
| news | Korean BERTScore F1 | 17 | 0.991725 | 0.988671 | 0.003054 | 0.000015 |
| news | chrF++ | 17 | 0.360251 | 0.223711 | 0.136541 | 0.000015 |
| news | Morpheme ROUGE-L | 17 | 0.540435 | 0.345062 | 0.195373 | 0.000015 |
| news | Extractive Coverage | 17 | 0.762022 | 0.633124 | 0.128898 | 0.000031 |
| news | Extractive Density | 17 | 3.892330 | 1.816353 | 2.075977 | 0.000015 |
| news | Novel-2 | 17 | 0.582175 | 0.761107 | -0.178932 | 0.000015 |
| news | Novel-4 | 17 | 0.814605 | 0.937177 | -0.122572 | 0.000015 |
| petition | Entity F1 | 9 | 0.648148 | 0.572751 | 0.075397 | 0.812500 |
| petition | Numeric/Date F1 | 12 | 0.660933 | 0.477611 | 0.183322 | 0.031250 |
| petition | BiNLI Support | 17 | 0.593782 | 0.184847 | 0.408935 | 0.001343 |
| petition | BiNLI Coverage | 17 | 0.455486 | 0.141920 | 0.313566 | 0.007904 |
| petition | BiNLI F1 | 17 | 0.485387 | 0.135587 | 0.349800 | 0.001678 |
| petition | Korean BERTScore Precision | 17 | 0.986935 | 0.982462 | 0.004472 | 0.000107 |
| petition | Korean BERTScore Recall | 17 | 0.987725 | 0.984248 | 0.003477 | 0.000076 |
| petition | Korean BERTScore F1 | 17 | 0.987329 | 0.983354 | 0.003975 | 0.000076 |
| petition | chrF++ | 17 | 0.226846 | 0.133378 | 0.093468 | 0.000015 |
| petition | Morpheme ROUGE-L | 17 | 0.390722 | 0.191461 | 0.199262 | 0.000031 |
| petition | Extractive Coverage | 17 | 0.605228 | 0.446914 | 0.158314 | 0.000153 |
| petition | Extractive Density | 17 | 2.881849 | 0.716249 | 2.165600 | 0.000031 |
| petition | Novel-2 | 17 | 0.757901 | 0.909948 | -0.152047 | 0.000046 |
| petition | Novel-4 | 17 | 0.918694 | 0.989902 | -0.071208 | 0.000655 |
| sns | Entity F1 | 1 | 1.000000 | 1.000000 | 0.000000 | NA |
| sns | Numeric/Date F1 | 7 | 0.714286 | 0.360544 | 0.353741 | 0.187500 |
| sns | BiNLI Support | 17 | 0.417281 | 0.046098 | 0.371183 | 0.000015 |
| sns | BiNLI Coverage | 17 | 0.378636 | 0.057149 | 0.321487 | 0.002090 |
| sns | BiNLI F1 | 17 | 0.358086 | 0.021943 | 0.336143 | 0.000015 |
| sns | Korean BERTScore Precision | 17 | 0.987891 | 0.982600 | 0.005291 | 0.000015 |
| sns | Korean BERTScore Recall | 17 | 0.988604 | 0.984486 | 0.004118 | 0.000015 |
| sns | Korean BERTScore F1 | 17 | 0.988247 | 0.983541 | 0.004706 | 0.000015 |
| sns | chrF++ | 17 | 0.212603 | 0.113586 | 0.099017 | 0.000015 |
| sns | Morpheme ROUGE-L | 17 | 0.396191 | 0.185960 | 0.210230 | 0.000015 |
| sns | Extractive Coverage | 17 | 0.620276 | 0.438396 | 0.181880 | 0.000015 |
| sns | Extractive Density | 17 | 1.458576 | 0.611262 | 0.847313 | 0.000015 |
| sns | Novel-2 | 17 | 0.764112 | 0.931849 | -0.167737 | 0.000015 |
| sns | Novel-4 | 17 | 0.952007 | 0.996196 | -0.044189 | 0.000655 |
