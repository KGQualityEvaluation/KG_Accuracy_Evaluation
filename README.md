# Evaluating Knowledge Graph Accuracy Powered by Optimized Human-machine Collaboration
A framework for graph accuracy evaluation considering both triple accuracy and entity linking accuracy. 

## Our Framework
There are four main components as follows.
- **IG Constructor** combines triples, linked entities, and dependencies together to build Inference Graphs (IGs);
- **Sampler** draws a subset of IGs by utilizing the sampling method S, which can be simple random sampling or stratified sampling, etc.;
- **Annotation Helper** assists the annotator by pre-computing proper triples and mentions to annotate and performing inference after receiving feedback from the annotator;
- **Collector** collects all the annotated IGs, calculates two estimates with their confidence intervals (CIs) using the annotated IGs, and determines when to stop the evaluation process.

To simplify the contruction stage, code of **IG Constructor** is placed in the *pre_processing* folder in *dataset/oneSpecficDataset*. Source code of our framework is placed in *code/framework*.

## Datasets
### YAGO & NELL
To use YAGO and NELL, one should download raw data (including triples, labels of triples and rules, collected by KGEval) from https://aclanthology.org/D17-1183/ and put them into *dataset/YAGO/raw_data* and *dataset/NELL/raw_data*. TGs constructed from raw data will be generated in *dataset/YAGO/data* and *dataset/NELL/data* after running:

YAGO:
```bash
cd dataset/YAGO
./pre_processing/pre_processing.sh
```

NELL:
```bash
cd dataset/NELL
./pre_processing/pre_processing.sh
```

The beginning and end of three stages (generating a graph from raw triples, processing the rules for future use and finally constructing TGs) will be shown in the screen.

### SYN-IGs
SYN-IGs are datasets of IGs generated randomly. To use SYN-IGs, one should generate them first using:

```bash
cd dataset/SYN-IG-10k
# generate TGs ignoring entity linking
./generate_graph/generate.sh "T"
# generate TGs with entity linking
./generate_graph/generate.sh "E"

cd dataset/SYN-IG-10m
# generate TGs ignoring entity linking
./generate_graph/generate.sh "T"
# generate TGs with entity linking
./generate_graph/generate.sh "E"

cd dataset/SYN-IG-100k
# generate TGs ignoring entity linking
./generate_graph/generate.sh "T"
# generate TGs with entity linking
./generate_graph/generate.sh "E"
```

## Run


