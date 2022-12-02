### CoCoLoT: Combining Complementary Trackers in Long-Term Visual Tracking

🏆 Winner of the [Visual Object Tracking VOT2021 Long-term Challenge](https://openaccess.thecvf.com/content/ICCV2021W/VOT/papers/Kristan_The_Ninth_Visual_Object_Tracking_VOT2021_Challenge_Results_ICCVW_2021_paper.pdf) (aka mlpLT)

Matteo Dunnhofer, Kristian Simonato, Christian Micheloni

Machine Learning and Perception Lab  
Department of Mathematics, Computer Science and Physics  
University of Udine  
Udine, Italy

##### Hardware and OS specifications
CPU Intel Xeon E5-2690 v4 @ 2.60GHz
GPU NVIDIA TITAN V 
320 GB of RAM
OS: Ubuntu 20.04  

#### VOT-LT test instructions
To run the VOT Challenge Long-term experiments please follow these instructions:

+ Clone the repository ``git clone https://github.com/matteo-dunnhofer/CoCoLoT``

+ Download the pre-trained weights files ``STARKST_ep0050.pth.tar``, ``super_dimp.pth.tar`` from [here](https://drive.google.com/drive/folders/1W_ePPy5HoLgcGbUE5Gh_0HZ034szvSiQ?usp=sharing) and put them in the folder ``CoCoLoT/ckpt/``

+ Move to the submission source folder ``cd CoCoLoT``

+ Create the Anaconda environment ``conda env create -f environment.yml``

+ Activate the environment ``conda activate CoCoLoT``

+ Install ninja-build ``sudo apt-get install ninja-build``

+ Edit the variable ``base_path`` in the file ``vot_path.py`` by providing the full-path to the location where the submission folder is stored,
and do the same in the file ``trackers.ini`` by substituting the paths ``[full-path-to-CoCoLoT]`` in line 9 and 13
 
+ Run ``python compile_pytracking.py``
 
+ Run the analysis by ``vot evaluate CoCoLoT`` 

+ Run the evaluation by ``vot analysis CoCoLoT``  

#### If you fail to run our tracker please write to ``matteo.dunnhofer@uniud.it``

#### An improved version of CoCoLoT exploiting Stark and KeepTrack is downloadable [here](http://data.votchallenge.net/vot2022/trackers/CoCoLoT-code-2022-04-28T08_08_35.527492.zip).

#### References

If you find this code useful please cite:

```
@INPROCEEDINGS{9956082,
    author={Dunnhofer, Matteo and Micheloni, Christian},
    booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
    title={CoCoLoT: Combining Complementary Trackers in Long-Term Visual Tracking}, 
    year={2022},
    pages={5132-5139},
    doi={10.1109/ICPR56361.2022.9956082}
}

@article{Dunnhofer2022imavis,
    title = {Combining complementary trackers for enhanced long-term visual object tracking},
    journal = {Image and Vision Computing},
    volume = {122},
    year = {2022},
    doi = {https://doi.org/10.1016/j.imavis.2022.104448}
}
```

The code presented here is built up on the following repositories:
 + [pytracking](https://github.com/visionml/pytracking)
 + [Stark](https://github.com/researchmm/Stark)
 + [LTMU](https://github.com/Daikenan/LTMU)
