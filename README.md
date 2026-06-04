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
##### Dataset: Split CIFAR-10, Feature extractor: Reduced Resnet-18, Memory size: 1000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar10 --backbone reduced --memory 1000 
  </code>
</pre>
##### Dataset: CORe-50, Feature extractor: Resnet-18, Memory size: 2000
<pre>
  <code id="code-snippet">
    python General_main.py --dataset core50 --backbone resnet18 --memory 2000
  </code>
</pre>
##### Dataset: Split CIFAR-100, Feature extractor: Resnet-50, Step: 2
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cifar100 --backbone resnet50 --step 2
  </code>
</pre>
##### Dataset: Split CUB-200, Feature extractor: Resnet-50, Step: 5
<pre>
  <code id="code-snippet">
    python General_main.py --dataset cub200 --backbone resnet50 --step 5
  </code>
</pre>
##### Case study: Split CIFAR-10, Feature extractor: Resnet-18, Step: 2
<pre>
  <code id="code-snippet">
    python General_main.py --case_study True --dataset cifar10 --backbone resnet18 --step 2
  </code>
</pre>
##### Runtime with respect to the number of seen classes: Split CIFAR-100, Feature extractor: Resnet-50, Step: 2, Memory size: 3000
<pre>
  <code id="code-snippet">
    python General_main.py --runtime True
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
      note         = {Submitted 2025}
    }  </code>
</pre>
