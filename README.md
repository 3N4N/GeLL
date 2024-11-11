# GeLL: Generalizable Log Parser using Large Language Model

The experiments and results for the paper submitted at [ISSTA2025][1].

## Abstract

Logs provide valuable insights into system runtime. Log parsing, which converts
semi-structured log data into structured log data, is often the first step in
automated log analysis. Existing log parsers generalize poorly in real-world
systems due to logs being generated from new log templates over time. Although a
large number of research has been done in log parsing, few have explored
generalization under new log templates. We propose GeLL, a novel LLM-based log
parser with generalization ability. GeLL employs an LLM to extract log templates
from raw logs. It generalizes under new log templates by utilizing a modified
test-time training approach by creating mutated logs from the new logs.
Experimental results on 16 benchmark datasets show that GeLL outperforms the
state of the art (SOTA) generalizable log parser, Log3T. GeLL offers better
grouping accuracy than Log3T in 10 out of 16 datasets and tied with Log3T in 3
out of the rest 6. GeLL also achieves a better performance compared to SOTA
online log parsers like Drain or Logram, obtaining 2.7% higher grouping accuracy
and 19.7% better edit distance than the SOTA baselines.

[1]: https://conf.researchr.org/home/issta-2025
