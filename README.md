# 4Ds

Direct Distillation between Different Domains

# Toolbox for 4Ds

This repository aims to provide a compact and easy-to-use implementation of our proposed 4Ds on standard image domain adaptation tasks (e.g., Office-Home, DomainNet). 

- Computing Infrastructure:
  - We use one NVIDIA V100 GPU for Office-Home experiments and use one NVIDIA A100 GPU for DomainNet experiments. The PyTorch version is 1.12.

- Please put the datasets (e.g. Office-Home, DomainNet) in the `./data/DA_dataset/`.
## Get the pretrained teacher models

```bash
# Office-Home
python train_teacher.py --batch_size 64 --epochs 120 --data_type officehome --model resnet34 --learning_rate 0.01 --epochs 120 --lr_decay_epochs 40,70,100 --weight_decay 5e-4 --trial 0 --gpu_id 0

# DomainNet
python train_teacher.py --batch_size 64 --epochs 120 --data_type domainnet --model resnet34 --learning_rate 0.01 --epochs 120 --lr_decay_epochs 40,70,100 --weight_decay 5e-4 --trial 0 --gpu_id 0
```

## Train the student models

```bash
# Office-Home 
# --A2C
python train_student.py --path_t ./save/teachers/models_DA/officehome/***.pth --model_t resnet34 --model_s resnet18 \
--learning_rate 0.01 --epochs 120 --lr_decay_epochs 40,70,100 --weight_decay 5e-4 --trial 0 --gpu_id 0 \
--data_type officehome --dataset Clipart --t_dataset Art

# DomainNet
# --C2P
python train_student.py --path_t ./save/teachers/models_DA/domainnet/***.pth --model_t resnet34 --model_s resnet18 \
--learning_rate 0.01 --epochs 120 --lr_decay_epochs 40,70,100 --weight_decay 5e-4 --trial 0 --gpu_id 0 \
--data_type domainnet --dataset Painting --t_dataset Clipart
```





