# animate-anyone-reproduction
reproduction of AnimateAnyone using SVD

### To Do list
- [x] piepline based on SVD
- [x] train V0.9 which can only generate 14 frames per ref-image
- [x] train animate-anyone like pipeline V1 which can generate arbitrary frames per ref-image
- [x] enhance face quality and time consistency(trick according to analyse animate anyone app cases)
- [x] release V1 inference code and model
---
**2024-02-25 update**
- V1 [checkpoint](https://modelscope.cn/models/lightnessly/animate-anyone-v1/summary) can be download now.
- We can not release V1.1 which is the latest version. But we will release V1.1 if we have V1.2 and so on.
- we also provide testcase to reproduce V1 result as below.
- the original result has bad quality on human face, so we use [simswap](https://github.com/neuralchen/SimSwap) to enhance face. More detials can be found in [issue](https://github.com/bendanzzc/AnimateAnyone-reproduction/issues/3).
- You should first download the SVD model, and then use the unet provided by us to replace the original unet.
- we find that the model has a certain degree of generalization on apperance and temporal consistency, but lacks the ability to generalize poses. So V1 can have a better performance on UBC pose.
- we only add 300 high quality videos to achieve V1.1 results, you can finetune by your own datset.
- we do not have any plans to release the training script but [svd-temporal-controlnet](https://github.com/CiaraStrawberry/svd-temporal-controlnet) may work.
---
 **2024-02-05 update**
- because of the [issue](https://github.com/bendanzzc/AnimateAnyone-reproduction/issues/4), we decide to release inference code in advance which is not well organized but works.
- as for postprocess of face, you can use any video face swap framework to do that. More details can be found in [issue](https://github.com/bendanzzc/AnimateAnyone-reproduction/issues/3).
- our inference code mainly baed on [svd-temporal-controlnet](https://github.com/CiaraStrawberry/svd-temporal-controlnet), you can also use training code to train your own model.
- our dataset is only UBC, but it can generalize to other simple domains. we will continue collecting high quailty video data.
---
 **2024-01-25 update**
- according to analyse animate anyone app cases, we find there may be some tricks instead of training model. so we will update the case which has better face quality with free training.
- the face enhance result shows below in the V1 part
---

### V1.1 animate-anyone ref-image case

https://github.com/bendanzzc/AnimateAnyone-reproduction/assets/26294676/e1efacad-b12e-4121-abcb-4276b3960a3b

### V1
**cross-domain case**

https://github.com/bendanzzc/AnimateAnyone-reproduction/assets/26294676/6add2e5f-a110-4513-adaa-ac24378971af

**with face enhance**

https://github.com/bendanzzc/AnimateAnyone-reproduction/assets/26294676/0af71e3f-623a-4f31-8fa7-d82ea86ae6c2

**ori result**

https://github.com/bendanzzc/AnimateAnyone-reproduction/assets/26294676/3bffc2db-6b46-4386-bed9-1d59dc7f62e1

https://github.com/bendanzzc/AnimateAnyone-reproduction/assets/26294676/027608d9-970b-4f3e-b47f-f95e7be8553c

---
### v0.9
https://github.com/bendanzzc/animate-anyone-reproduction/assets/26294676/57b65b96-1391-47b5-81c0-2b25700c5aaa

https://github.com/bendanzzc/animate-anyone-reproduction/assets/26294676/2d0ebb99-e632-46b6-9780-f2443361977a


