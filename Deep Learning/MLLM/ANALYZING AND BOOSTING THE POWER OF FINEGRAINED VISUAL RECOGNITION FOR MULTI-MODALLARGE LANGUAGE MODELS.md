### ANALYZING AND BOOSTING THE POWER OF FINEGRAINED VISUAL RECOGNITION FOR MULTI-MODALLARGE LANGUAGE MODELS

#### Motivation

MLLM still struggle with fine-grained visiual recognition(FGVR), with aims to identify subordinates-level categories from images.

#### Preliminary

The three quintessential capabilities of MLLM:

1. Object information extraction
2. Category knowledge reserve
3. Object-category alignment

They analyzed the representation space of MLLMs and their cosreponding visual language models, the conclusion is that:

1. Object information lost exists berween VLMs and MLLMs but is not the bottleneck.
2. Category knowledge is relative sufficient, but category names cannot fully capture the senmatics.
3. Misalignment between the visual object and category name leads to underperformance.

#### Notions

- Assuming an Image $I_i$ containing an Object $O_i$ is processed by vision encoder $V_\alpha$ and learnable modality connector $F_\beta$ to be transformed into a visual object token sequence of length $m$: $S_o^{i}=[o_1^i, o_2^i, \cdots, o_m^{i}]$.

- Input category name in textual modality $C_i$ is passed through an embedding layer of LLM to obtain the category embedding sequence of length $n$: $S_c^i=[c_1^i, c_2^2, \cdots, c_n^i]$. 

- The object embedding sequence $S_o^{i}$ and category embedding sequence $S_c^i$ are individually passed through LLM Layers $L_\theta$ to obtain the output from the last layers.
  $$
  H_o^i=L_\theta(S^i_o) \\
  H_c^i=L_\theta(S^i_c)
  $$
  Afterward, they select two ways to represent the global semantics of output sequence:

  1. last token embedding
  2. average of the token embedding

#### Analyzed

##### Object Information Extraction

<img src="../resource/image-20250130014338157.png" alt="image-20250130014338157" style="zoom:67%;" />

- The output object token from the vision encoder perserves discriminative information for classification as shown in Figure2(a).
- Various objects belonging to the same subordinate-level categories can still cluster together and distance each other as shown in Figure2(b).