# Enhancing Transformer RNNs with Multiple Temporal Perspectives

Authors: [Razvan-Gabriel Dumitru](mailto:razvandumm@gmail.com?subject=[GitHub]TemporalRNNs), [Darius Peteleaza](mailto:peteleaza.darius@gmail.com?subject=[GitHub]TemporalRNNs) & [Mihai Surdeanu](mailto:msurdeanu@arizona.edu?subject=[GitHub]TemporalRNNs)

If you wish to contact any of us for any reason, please use the above click-able email links.

The paper was published at the ```ICML 2024 - Next Generation of Sequence Modeling Architectures Workshop - 26 July 2024```.

The detailed paper can be found at https://arxiv.org/abs/2402.02625. Please cite our work if you use it.

## Abstract
We introduce the concept of multiple temporal perspectives, a novel approach applicable to Recurrent Neural Network (RNN) architectures for enhancing their understanding of sequential data. This method involves maintaining diverse temporal views of previously encountered text, significantly enriching the language models' capacity to interpret context. To show the efficacy of this approach, we incorporate it into the Receptance Weighted Key Value (RWKV) architecture, addressing its inherent challenge of retaining all historical information within a single hidden state. Notably, this improvement is achieved with a minimal increase in the number of parameters --even as little as $0.04\%$ of the original number of parameters. Further, the additional parameters necessary for the multiple temporal perspectives are fine-tuned with minimal computational overhead,
avoiding the need for a full pre-training. The resulting model maintains linear computational complexity during prompt inference, ensuring consistent efficiency across various sequence lengths. The empirical results and ablation studies included in our research validate the effectiveness of our approach, showcasing improved performance across multiple benchmarks.

## Citation
If you use our work in your research, please cite our paper:

```
Dumitru, R. G., Peteleaza, D., & Surdeanu, M. (2024). Enhancing Transformer RNNs with Multiple Temporal Perspectives. arXiv preprint arXiv:2402.02625.
```

### BibTeX

```
@misc{dumitru2024enhancingtransformerrnnsmultiple,
      title={Enhancing Transformer RNNs with Multiple Temporal Perspectives}, 
      author={Razvan-Gabriel Dumitru and Darius Peteleaza and Mihai Surdeanu},
      year={2024},
      eprint={2402.02625},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.02625}, 
}
```
