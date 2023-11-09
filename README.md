
# Cas6D: 6-DoF Promptable Pose Estimation of Any Object, in Any Scene, with One Reference

[[Paper]](https://arxiv.org/abs/2306.07598) [[Project]](https://github.com/paulpanwang/Cas6D)  



We present a new cascade framework named Cas6D for few-shot 6DoF pose estimation that is generalizable and uses only RGB images.


## Evaluate all components together.
```shell
# Evaluate on the object TFormer from the GenMOP/LINEMOD dataset
python eval.py --cfg configs/cas6d_train.yaml 

```


## Acknowledgement

We would like to thank [Gen6D](https://github.com/liuyuan-pal/Gen6D) authors for open-sourcing their implementations.


## Citation

If you find this repo is helpful, please consider citing:
```bibtex
@article{pan2023learning,
  title={Learning to Estimate 6DoF Pose from Limited Data: A Few-Shot, Generalizable Approach using RGB Images},
  author={Pan, Panwang and Fan, Zhiwen and Feng, Brandon Y and Wang, Peihao and Li, Chenxin and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2306.07598},
  year={2023}
}

```



