# pytorch--CutMix (pytorch)
- Implementation with torch
- Other code for model and normalization available at https://github.com/LEEYEONSU/pytorch--SENet

---

- For training

  ~~~
  python main.py 
  --print_freq 32 --save_dir ./save_model/ --save_every 10
  --lr 0.1 --weight_decay 1e-4 --momentum 0.9 
  --Epoch 500 --batch_size 128 --test_batch_size 100 
  --cutout False --n_masks 1 --length 16 
  --normalize batchnorm
  --alpha 1.0 --cutmix_prob 1.0  # For Cutmix
  ~~~

---

##### Result 

- Comparison between SE + resnet-32 + batchnorm **&** SE + resnet-32 + batchnorm + cutmix 

  |               | [SE + resnet-32 + batchnorm](https://github.com/LEEYEONSU/pytorch--SENet) | SE + resnet-32 + batchnorm + [cutmix](https://github.com/LEEYEONSU/pytorch--CutMix/blob/main/utils/cutmix.py) |
  | ------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | top - 1 error |                             4.76                             |                             4.50                             |




##### Data

- CIFAR -10 

##### Model

- SE (Squeeze and Excitation Network with Resnet-32) 

  