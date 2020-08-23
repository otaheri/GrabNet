# GrabNet
Generating realistic hand mesh grasping unseen 3D objects (ECCV 2020)

# Coming Soon ...

[![report](https://img.shields.io/badge/arxiv-report-red)](https://grab.is.tue.mpg.de)

![GRAB-Teaser](images/teaser.png)
[[Paper Page](https://grab.is.tue.mpg.de)] [[Paper](https://ps.is.mpg.de/publications/grab-2020) ]
[[Supp. Mat.](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490562-supp.pdf)]

[GrabNet](http://grab.is.tue.mpg.de) is a generative model for 3D hand grasps. Given a 3D object mesh, GrabNet 
can predict several hand grasps for it. GrabNet has two succesive models, CoarseNet (cVAE) and RefineNet.
It is trained on a subset (only right hand grasps) of [GRAB](http://grab.is.tue.mpg.de) dataset.
For more details please refer to the [Paper](https://ps.is.mpg.de/publications/grab-2020) or the [project website](http://grab.is.tue.mpg.de).

Below you can see some generated results from GrabNet:

| Binoculars | Mug |Camera | Toothpaste|
| :---: | :---: |:---: | :---: |
| ![Binoculars](images/binoculars.gif)|![Mug](images/mug.gif)|![Camera](images/camera.gif)|![Toothpaste](images/toothpaste.gif)|



Check out the YouTube videos below for more details.

| Short Video | Long Video |
| :---: | :---: |
|  [![ShortVideo](images/short.png)](https://youtu.be/VHN0DBUB4H8) | [![LongVideo](images/long.png)](https://youtu.be/s5syYMxmNHA) | 


## Table of Contents
  * [Description](#description)
  * [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Examples](#examples)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)



## Description

This implementation:

- Can run GrabNet on unseen objects to generate several grasps
- Includes the code to retrain GrabNet with different configuration
- Test results on test objects


## Getting started
Inorder to use GrabNet model please follow the below steps:

- Download the GrabNet dataset from [this website](http://grab.is.tue.mpg.de) and put it in the following structure:
```bash
    GRAB
    ├── data
    │    │
    │    ├── bps.npz
    │    └── obj_info.npy
    │    └── sbj_info.npy
    │    │
    │    └── [split_name] from (test, train, val)
    │          │
    │          └── frame_names.npz
    │          └── grabnet_[split_name].npz
    │          └── data
    │                └── s1
    │                └── ...
    │                └── s10
    └── tools
         │
         ├── object_meshes
         └── subject_meshes
```
- Download the GrabNet models from the [paper](https://grab.is.tue.mpg.de) and put the models folder in the cloned repository.
- Go to the [MANO website](https://mano.is.tue.mpg.de) to download MANO models (skip this part if you already followed this for GRAB dataset).
- Follow the instalation steps for this repo in the next section.


## Requirements
This package has the following requirements:

* [Pytorch>=1.1.0](https://pytorch.org/get-started/locally/) 
* Python >=3.6.0
* [pytroch3d >=0.2.0](https://pytorch3d.org/) 
* [MANO](https://github.com/otaheri/MANO) 
* [meshviewer](https://github.com/MPI-IS/mesh) (for visualization)
* [bps_torch](https://github.com/otaheri/bps_torch) 

## Installation

To install the dependencies please follow the next steps:

- Clone this repository and install the requirements: 
    ```Shell
    git clone https://github.com/otaheri/GrabNet
    ```
- Install the dependencies by the following command:
    ```
    pip install -r requirements.txt
    ```
- Install the meshviewer from [this repo](https://github.com/MPI-IS/mesh)


## Examples

After installing the *GrabNet* package, dependencies, and downloading the data and the models from
 mano website, you should be able to run the following examples:


- #### Generate several grasps for new objects
    
    ```Shell
    python grabnet/tests/grabnew_objects.py --obj-path $NEW_OBJECT_PATH \
                                            --rhm-path $MANO_MODEL_FOLDER \
                                            --data_path $PATH_TO_GRABNET_DATA
    ```

- #### Generate grasps for test data and compare to GT
    
    ```Shell
    python grabnet/tests/test.py     --rhm-path $MANO_MODEL_FOLDER \
                                     --data_path $PATH_TO_GRABNET_DATA
    ```

- #### Train GrabNet with new configurations 
    
    To retrain GrabNet with new configuration, please use the following code.
    
    ```Shell
    python train.py  --work-dir $SAVING_PATH \
                    --rhm-path $MANO_MODEL_FOLDER \
                    --data_path $PATH_TO_GRABNET_DATA
    ```
    
- #### Get the GrabNet evaluation errors on the dataset 
    
    ```Shell
    python eval.py     --rhm-path $MANO_MODEL_FOLDER \
                       --data_path $PATH_TO_GRABNET_DATA
    ```



## Citation

```
@inproceedings{GRAB:2020,
  title = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  author = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2020},
  url = {https://grab.is.tue.mpg.de}
}
```

## License
Software Copyright License for **non-commercial scientific research purposes**.
Please read carefully the following [terms and conditions](https://github.com/otaheri/GRAB/blob/master/LICENSE) and any accompanying documentation
before you download and/or use the GRAB data, model and software, (the "Data & Software"),
including 3D meshes (body and objects), images, videos, textures, software, scripts, and animations.
By downloading and/or using the Data & Software (including downloading,
cloning, installing, and any other use of the corresponding github repository),
you acknowledge that you have read these terms and conditions, understand them,
and agree to be bound by them. If you do not agree with these terms and conditions,
you must not download and/or use the Data & Software. Any infringement of the terms of
this agreement will automatically terminate your rights under this [License](./LICENSE).


## Acknowledgments

Special thanks to [Mason Landry](https://github.com/soubhiksanyal) for his invaluable help with this project.

We thank S. Polikovsky, M. Hoschle (MH) and M. Landry (ML)
for the MoCap facility. We thank F. Mattioni, D. Hieber, and A. Valis for MoCap
cleaning. We thank ML and T. Alexiadis for trial coordination, MH and F. Grimminger
for 3D printing, V. Callaghan for voice recordings, J. Tesch for renderings, and Benjamin Pellkofer for the IT support. 
We thank Sai Kumar Dwivedi and Nikos Athanasiou for proofreading.
## Contact
The code of this repository was implemented by [Omid Taheri](https://ps.is.tuebingen.mpg.de/person/otaheri) and [Nima Ghorbani](https://ps.is.tuebingen.mpg.de/person/nghorbani).

For questions, please contact [grab@tue.mpg.de](grab@tue.mpg.de).

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](ps-licensing@tue.mpg.de).

