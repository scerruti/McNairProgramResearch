# Results
We compare to state-of-the-art monolithic VLMs and Program Synthesis approaches. For each benchmark, we breakdown performance for numeric, yes/no and multiple-choice answers and report total average. To evaluate program correctness for program synthesis methods, we report oracle variants which substitute vision specialists with oracle ones.
1. [Comparison on Omni3D-Bench](#omni3d-bench)
2. [Comparison on CLEVR](#clevr)
3. [Comparison on VQA](#gqa)

## Omni3D-Bench <a name="omni3d-bench"></a>
We report performance on 500 questions from the Omni3D-Bench dataset. The first table compares VADAR with monolithic VLMs, the second table compares VADAR to other program synthesis methods. The last table compares oracle performance to other program synthesis methods on a 55Q subset of Omni3D-Bench. We report accuracy for all columns except numeric (other), where we report [Mean Relative Accuracy (MRA)](https://vision-x-nyu.github.io/thinking-in-space.github.io/)

1. Comparison with monolithic VLMs

| Method    | numeric (count) | numeric (other) | y/n  | multi-choice |Total |
|------------------|-----------------|-----------------|------|--------------|------|
| GT4o             | 28.1            | 35.5            | 66.7 | 57.2         | 42.9 |
| Claude3.5-Sonnet | 22.4            | 20.6            | 62.2 | 50.6         | 32.2 |
| Llama3.2         | 24.3            | 19.3            | 47.5 | 27.4         | 25.6 |
| Gemini1.5-Pro    | 25.2            | 28.1            | 46.2 | 37.6         | 32.0 |
| Gemini1.5-Flash  | 24.3            | 27.6            | 51.1 | 52.9         | 35.0 |
| Molmo            | 21.4            | 21.7            | 29.3 | 41.2         | 26.1 |
| SpaceMantis      | 20.0            | 21.7            | 50.6 | 48.2         | 30.3 |
| **VADAR**        | 21.7            | 35.5            | 56.0 | 57.6         | 40.4 |

2. Comparison with program synthesis methods.

| Method       | numeric (count) | numeric (other) | y/n | multi-choice |Total|
|--------------------|-----------------|-----------------|-----|--------------|-----|
| ViperGPT           | 20.0            | 15.4            |56.0 | 42.4         | 33.5|
| VisProg            | 02.9            | 00.9            |54.7 | 25.9         | 21.1|
| **VADAR**          | 21.7            | 35.5            |56.0 | 57.6         | 40.4|

3. Oracle comparison with program synthesis methods.

| Method             | numeric | y/n   | multi-choice | Total |
|--------------------|---------|-------|--------------|-------|
| ViperGPT + oracle  | 48.5    | 66.7  | 49.3         | 54.9  |
| VisProg + oracle   | 60.6    | 68.5  | 66.7         | 66.0  |
| **VADAR + oracle** | 89.5    | 100.0 | 94.1         | 94.4  |



## CLEVR <a name="clevr"></a>
We report performance on 1155 questions from the CLEVR dataset. The first table compares VADAR with monolithic VLMs, the second table compares VADAR to other program synthesis methods. 

1. Comparison with monolithic VLMs

| Method       | numeric | y/n  | multi-choice | Total |
|--------------------|---------|------|--------------|-------|
| GPT4o              | 52.3    | 63.0 | 60.0         | 58.4  |
| Claude3.5-Sonnet   | 44.7    | 61.4 | 72.2         | 58.9  |
| Llama3.2-11B       | 34.6    | 45.6 | 49.0         | 42.8  |
| Gemini1.5-Pro      | 44.9    | 59.7 | 67.0         | 56.9  |
| Gemini1.5-Flash    | 43.1    | 58.8 | 56.8         | 52.8  |
| Molmo-7B           | 11.0    | 42.6 | 51.4         | 34.4  |
| SpaceMantis        | 14.5    | 52.9 | 32.3         | 33.2  |
| **VADAR**          | 53.3    | 65.3 | 40.8         | 53.6  |
| **VADAR + oracle** | 82.4    | 85.4 | 81.0         | 83.0  |

2. Comparison with program synthesis methods.

| Method       | numeric | y/n  | multi-choice | Total |
|--------------------|---------|------|--------------|-------|
| ViperGPT           | 20.5    | 43.4 | 13.4         | 26.2  |
| VisProg            | 16.7    | 48.4 | 28.3         | 31.2  |
| **VADAR**          | 53.3    | 65.3 | 40.8         | 53.6  |
| ViperGPT + oracle  | 38.5    | 57.8 | 30.2         | 40.6  |
| VisProg + oracle   | 25.3    | 52.5 | 41.8         | 39.9  |
| **VADAR + oracle** | 82.4    | 85.4 | 81.0         | 83.0  |

## GQA <a name="gqa"></a>
We report performance on select methods on a subset of the testdev split of the GQA dataset. GQA focuses primarily on object appearance, not 3D spatial reasoning.

| Method       | Accuracy (%)      |
|--------------|------|
| GT4o         | 54.9 |
| ViperGPT     | 42.0 |
| VisProg      | 46.9 |
| **VADAR**    | 46.1 |
