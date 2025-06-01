CDSANet: A CNN-ViT-Attention Network for Ship Instance Segmentation in Visible-Light Images

Ship instance segmentation plays a crucial role in maritime applications such as autonomous navigation, but it remains a challenging task because of the complex oceanic environment and significant variability in ship appearances. This paper proposes CDSANet, a novel single-stage instance segmentation network that integrates convolutional operations, Vision Transformers, and attention mechanisms. The backbone incorporates the CTMvitbv3 module to enhance global-local feature representation, while the neck employs DOWConv with dynamic weights for improved handling of challenging backgrounds and objects at multiple scales. Additionally, SIoU is adopted to enhance localization accuracy and rotational robustness, and CBAM is used to reinforce attention to critical features. To address the scarcity of visible-light ship segmentation data, we propose a new visible-light dataset, termed VLRSSD. Experiments conducted on the MarShipInsSeg and VLRSSD datasets demonstrate that CDSANet achieves state-of-the-art performance, with AP scores of 47.6% and 75.2%, respectively, representing improvements of 1.0% and 1.1% over the YOLOv8 baseline. In terms of efficiency, CDSANet outperforms existing mainstream instance segmentation methods in inference speed (130.3 FPS), parameter count (253M), and computational cost (111.4 GFLOPs), highlighting its strong potential for real-world deployment. Overall, CDSANet presents a favorable trade-off between segmentation accuracy and computational efficiency. The source code is publicly available at: https://github.com/RainbowSugar1/CDSANet.

Key requirements:
  PyTorch ==2.0.0
  CUDA ==12.6
  mmcv==1.6.2
  numpy==1.23.2
  python==3.18.7
Key Components:
  




