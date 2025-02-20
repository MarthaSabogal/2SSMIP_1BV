# Two-Stage Stochastic Mixed Integer Program
First Stage: Binary variables

Second Stage: One binary variable and continuous variables

# Code
IL_V1: Only using integer L-shaped cuts and solving each scenario subproblem as a MIP.

IL_V2: Only using integer L-shaped cuts and solving each scenario subproblem as 2LPs.

AIL_V1: Alternating integer L-shaped cuts (Benders + integer L-shaped cuts) and solving each scenario subproblem as a MIP.


Reusing information from Benders

AIL_V2: Alternating integer L-shaped cuts (Benders + integer L-shaped cuts) and solving each scenario subproblem as a MIP.

AIL_V3: Alternating integer L-shaped cuts (Benders + integer L-shaped cuts) and solving each scenario subproblem as a MIP.

AIL_V4: Alternating integer L-shaped cuts (Benders + integer L-shaped cuts) and solving each scenario subproblem as a MIP.

AIL_V5: Alternating integer L-shaped cuts (Benders + integer L-shaped cuts) and solving each scenario subproblem as 2LPs.


DC: Only using disjunctive cuts.

ADC_V1: Alternating disjunctive cuts (Benders + disjunctive cuts).

ADC_V2: Alternating disjunctive cuts (Benders + disjunctive cuts). Reusing information from Benders

AILDC_V1: Benders + integer L-shaped + disjunctive cuts.

AILDC_V2: Benders + integer L-shaped + disjunctive cuts. Reusing information from Benders

BC: Only using bilinear cuts.



