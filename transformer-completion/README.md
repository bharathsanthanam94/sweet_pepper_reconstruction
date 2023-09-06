# Shape Completion with MaskFormer

Currently training on complete point clouds

### Install

`pip3 install -U -e .`

### Usage

- train `python mask_ps/scripts/train_model.py`
- test ?
- pretrain backbone ?

### TODOs
- [x] make it work with batch size > 1
- [x] normalization losses as in template matching
- [x] testing script
- [x] check if all fruits are centered
- [ ] play around with unnormalized distances in knn up
- [ ] encode distance from template point as feature
- [x] fill holes in post processing
- [ ] fix bug in leaves dataloader, 0 is stem, we don't want stems
- [ ] do some kind of test/train/val split in leaves dataloader
- [x] chamfer distance 1m when empty prediction to allow for monitor 
- [x] automatic dataloader=train if overfit = True
- [x] iterative template deformation
- [x] put self attention in decoder
- [ ] attention instead of knn interpolation
- [ ] pretrain backbone or add aux loss
- [x] change scheduler lr
- [x] save best model not last
- [ ] use multi-resolution features
- [ ] predict fruit pose