# fastforks: the model settings of upstream config.py `Config` that the BiRefNet / BiRefNet_lite checkpoints were
# trained with (HF ZhengPeng7/BiRefNet birefnet.py); paths, data and training settings removed, the backbone is an
# argument (upstream: an index into its backbone list).
class Config():
    def __init__(self, bb='swin_v1_l') -> None:
        self.batch_size = 4  # fastforks: only `> 1` matters here (BatchNorm layers); HF birefnet.py: 4

        # MODEL settings
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.dec_blk = ['BasicDecBlk', 'ResBlk'][0]

        # Backbone settings
        self.bb = bb
        self.freeze_bb = 'dino_v3' in self.bb
        self.lateral_channels_in_collection = {
            'swin_v1_l': [1536, 768, 384, 192], 'swin_v1_b': [1024, 512, 256, 128],
            'swin_v1_s': [768, 384, 192, 96], 'swin_v1_t': [768, 384, 192, 96],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # MODEL settings - inactive
        self.lat_blk = ['BasicLatBlk'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.auxiliary_classification = False       # Only for DIS5K, where class labels are saved in `dataset.py`.
