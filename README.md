
# Generating 3D faces using Convolutional Mesh Autoencoders

The repository reproduces experiments as described in the paper of "Generating 3D faces using Convolutional Mesh
Autoencoders (CoMA)".

A. Ranjan, T. Bolkart, S. Sanyal, and M. J. Black. [Generating 3d faces using convolutional mesh autoencoders](https://arxiv.org/abs/1807.10267) (ECCV 2018)

This paper proposed to learn non-linear representations of a face using spectral convoultions ([ChebyNet](https://arxiv.org/abs/1606.09375)) on a mesh surface. More importantly, they introduced up- and down- sampling operations as core components of Mesh Autoencoders, enabling the model to learn hierarchical mesh representations that capture expressions at multiscales.  

The results reported in the paper show a high preproducibility. Follow the same network architecture with only slightly difference w.r.t the choice of activation function and optimizing hyperparamters, our implmentation already gives **better results** than those shown in the paper. For instance, the results of the interpolation experiment (with 30,059 parameters) are <img src="svgs/befa881c50e2ef3fbd6e13a34d5b4f19.svg" align=middle width=194.967795pt height=22.381919999999983pt/>, <img src="svgs/06c33cc4b488a118711841b94d6b57f9.svg" align=middle width=151.319355pt height=22.745910000000016pt/>, compared to those reported in the paper (with 33,856 paramters) with <img src="svgs/821bfb70c189abe5923295ad516ade68.svg" align=middle width=194.967795pt height=22.381919999999983pt/>, <img src="svgs/db779979f43cb40665a81fc0b8dc7315.svg" align=middle width=151.319355pt height=22.745910000000016pt/>.

## ChebyShev Convolution
Recall the chebyshev graph convolutional operator from the paper "[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375)":
<p align="center"><img src="svgs/c6170f565cc371df15e12c562f75ca3c.svg" align=middle width=132.60241499999998pt height=48.153435pt/></p>

where <img src="svgs/930dee0f33b066e9002f9c339d3d22d4.svg" align=middle width=14.927055000000001pt height=22.745910000000016pt/> is a learnable parameter. The Chebyshev polynomial <img src="svgs/8a1ae15af22fb8f3bc20c8e74486debc.svg" align=middle width=228.84559499999997pt height=31.056300000000004pt/>, and <img src="svgs/dc8dc5a2f03a5937263a8b1b75664767.svg" align=middle width=11.145420000000001pt height=31.056300000000004pt/> is a scaled and normalized Laplacian defined as <img src="svgs/51ea0560a4cd83df374dd27346764491.svg" align=middle width=244.76479500000002pt height=31.056300000000004pt/>. In our implementation, we tacitly assume <img src="svgs/8e52866581d9d98977475c483ebae6d5.svg" align=middle width=66.6831pt height=22.745910000000016pt/> which is chosen from the official [ChebyConv repository](https://github.com/mdeff/cnn_graph/blob/c4d2c75d1807a1d1189b84bd6f4a0aafca5b8c53/lib/models.py#L885).

## Sampling Operation
This paper performs the in-network down- and up-sampling operations on mesh with the precomputed sampling matrices. The down-sampling matrix D is obtained from iteratively contracting vertex pairs that maintain surface error approximations using quadric matrics, and the up-sampling matrix U is obtained from including barycentric coordinates of the vertices that are discarded during the downsampling. It can be simply defined as:
<p align="center"><img src="svgs/495643a79495f6d3ce50d4936365a15e.svg" align=middle width=77.33054999999999pt height=13.156093499999999pt/></p>

where the *sparse* sampling matrix <img src="svgs/9180e00e196978aa798f62467e585afa.svg" align=middle width=80.2329pt height=30.950700000000015pt/> and node feature matrix <img src="svgs/281195f9409164ae6087fe6f0131dcb6.svg" align=middle width=98.84704500000001pt height=27.598230000000008pt/>.

The real magic of our implemtation happens in the body of ``models.networks.Pool``.  Here, we need to perform batch matrix multiplication on GPU w.r.t the sampling operation described above. Because dense matrix multiplication is really slow, we implement **sparse batch matrix multiplication** via scattering add node feature vectors corresponds to *cluster nodes* across a batch of input node feature matrices.

## Installation
The code is developed using Python 3.6 on Ubuntu 16.04. The models were trained and tested with NVIDIA 2080 Ti.
* [Pytorch](https://pytorch.org/) (1.3.0)
* [Pytorch Geometric](https://github.com/rusty1s/pytorch_geometric) (1.3.0)
* [OpenMesh](https://github.com/nmaxwell/OpenMesh-Python) (1.1.3)
* [MPI-IS Mesh](https://github.com/MPI-IS/mesh): We suggest to install this library from the source.
* [tqdm](https://github.com/tqdm/tqdm)

## Interpolation Experiment
Following the same split as described in the paper, the dataset is split in training and test samples with a ratio of 9:1. Run the script below to train and evaluet the model. The checkpoints of each epoch is saved in the corresponding output folder (specifed by the vairable ``exp_name``).  After training, it outputs the result of the "Mean Error with the Standard Deviation" as well as "Median Error", which are saved in the file ``euc_error.txt``.
```bash
bash train_interpolation.sh
```

## Extrapolation Experiment
To reproduce the extrapolation experiment, you should specify the test expression as described in the paper. We provide the vaiable ``test_exp`` to explictly specify the test expression. Run the script below to have a glance of the results.
```bash
bash train_extrapolation.sh
```

## Data
To create your own dataset, you have to provide data attributes at least:
- `data.x`: Node feature matrix with shape `[num_nodese, num_node_features]` and type `torch.float`.
- `data.edge_index`: Graph connectivity in COO format with shape `[2, num_edges]` and type `torch.long`. Note that to use this framework, the graph connectivity across all the meshes should be the same.

where `data` is inherited from `torch_geometric.data.Data`. Have a look at the classes of `datasets.FAUST` and `datasets.CoMA` for an example.

Alternatively, you can simply create a regular python list holding `torch_geometric.data.Data` objects

## Citation
Please cite [this paper](https://arxiv.org/abs/1807.10267) if you use this code in your own work:
```
@inproceedings{ranjan2018generating,
  title={Generating 3D faces using convolutional mesh autoencoders},
  author={Ranjan, Anurag and Bolkart, Timo and Sanyal, Soubhik and Black, Michael J},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={704--720},
  year={2018}
}
```
