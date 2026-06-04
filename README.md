# QDOL: Incremental Quality-Diversity Exemplar Selection for Online Task-Free Continual Learning

## Abstract
Online task-free class-incremental learning remains challenging due to catastrophic forgetting, unknown task boundaries, and strict memory constraints. Existing rehearsal-based methods often rely on repeated optimization or complex update procedures that are less suitable for true online learning. In this work, we propose QDOL, a simple and efficient rehearsal-based framework that combines quality-diversity exemplar selection with a training-free nearest class mean classifier. QDOL incrementally maintains representative and diverse exemplars under a fixed memory budget through a quality-weighted greedy orthogonal selection strategy and an efficient online replacement mechanism. The proposed framework processes each sample only once, requires no classifier retraining, and supports fully task-free online updates. Extensive experiments on CIFAR-10, CIFAR-100, CUB-200, and CORe-50 demonstrate that QDOL achieves competitive or state-of-the-art performance with efficient online inference and update behavior under strict memory constraints.

**Keywords:** Exemplar-based, Online task-free, Continual learning.
## Dataset
- Split CIFAR-10
- Split CIFAR-100
- Split CUB-200
- CORe-50
## Feature extractor
- Resnet-18
- Resnet-50
## Sample commands to run QDOL
##### Dataset: Split CIFAR-10, Budget (Memory size): 1000, Method: QDOL
<pre>
  <code id="code-snippet">
    python Main.py --dataset cifar10 --budget 1000 --method qdol
  </code>
</pre>
##### Dataset: CORe-50, Budget: 3000, Method: Herding
<pre>
  <code id="code-snippet">
    python Main.py --dataset core50 --budget 2000 --method herding
  </code>
</pre>
##### Dataset: Split CIFAR-100, Budget: 3000, Step: 5, Method: QDOL offline
<pre>
  <code id="code-snippet">
    python Main.py --dataset cifar100 --step 5 --method qd_off
  </code>
</pre>
##### Dataset: Split CUB-200, Budget: 3000, Step: 5, Method: DPP
<pre>
  <code id="code-snippet">
    python Main.py --dataset cub200 --step 5 --method dpp
  </code>
</pre>

## Citation
If you use this code in your research, please cite the following relevant work:
<pre>
  <code id="code-snippet">
    @article{tutc_QDOL,
      author       = {Cong Tu Tran, Thanh Tuan Nguyen, Thanh Phuong Nguyen, and Nadège Thirion-Moreau},
      title        = {QDOL: Incremental Quality-Diversity Exemplar Selection for Online Task-Free Continual Learning},
      conference   = {ACIVS 2026},
      note         = {Submitted 2026}
    }  </code>
</pre>
