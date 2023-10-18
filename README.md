This project introduces PosePro, a cost-effective and user-friendly system that addresses this issue. PosePro allows users to upload a photo of their desired pose, referred to as the ground truth pose, and stands in front of a camera. The system uses real-time pose detection to compare the user’s pose with the ground truth pose, displaying the similarity score on the screen. Users can also select poses from existing datasets or create their own.
The system incorporates powerful deep neural networks for image classification and utilizes MediaPipe for pose landmark detection. The architecture encompasses two main approaches: in the first case, users upload a reference pose for comparison, while in the second case, a pre-trained model performs classification for comparison. For the latter, a dataset of yoga poses was employed for training, with careful data preprocessing. A variety of deep learning models, including ResNet-18, ResNet-50, EfficientNet, and Vision Transformer (ViT), were explored for classification. The evaluation phase revealed that EfficientNet trained with an extended dataset of 83 classes, including unrelated im- ages for a more realistic scenario, achieved the best performance. The result is a universal pose correction platform that empowers users to improve their workout performance safely and effectively.

References

[1] M. Verma, S. Kumawat, Y. Nakashima, and S. Raman, “Yoga-82: A new dataset for fine-grained classification of human poses,” in IEEE/CVF Conference on Computer Vi- sion and Pattern Recognition Workshops (CVPRW), 2020, pp. 4472–4479. 2, 3

[2] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE confer- ence on computer vision and pattern recognition, 2016, pp. 770–778. 2

[3] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International confer- ence on machine learning. PMLR, 2019, pp. 6105–6114. 2

[4] N. Raw, “Fine-tune vision transformer (vit) for image clas- sification,” https://huggingface.co/blog/fine-tune-vit/, 2021. 2,5

[5] Wesley, “3d human pose classification using medi- apipe and pointnet,” Medium, 2022. [Online]. Avail- able: https://medium.com/@birddropping/3d-human-pose-classification-using-mediapipe-and-pointnet-cb2dd955c273

[6] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, “Blazepose: On-device real- time body pose tracking,” 2020. 3

[7] S. Raj, “Indian food image classification fine-tuning,”
https : / / github . com / rajshah4 / huggingface - demos / blob / main / FoodApp / Indian food image classification fine - tuning.ipynb, 2021. 5

[8] E.Odemakinde,“Humanposeestimationwithdeeplearning – ultimate overview in 2023,” Website, 2023. [Online]. Available: https://viso.ai/deep-learning/pose-estimation- ultimate- overview/

[9] F. B. Ashraf, M. U. Islam, M. R. Kabir, and J. Uddin, “Yonet: A neural network for yoga pose classification,” SN Computer Science, vol. 4, no. 2, p. 198, 2023. [Online]. Available: https://doi.org/10.1007/s42979-022-01618-8

[10] Logeswaran123, “MrPose,” https : / / github . com / Logeswaran123/MrPose, 2022.

[11] A. Bearman and C. Dong, “Human pose estimation and activity classification using convolutional neural networks,” CS231n Course Project Reports, 2015.

[12] A. Vij, “Push-ups with python! (mediapipe + open,” Feb 2022. [Online]. Available: https://aryanvij02.medium.com/ push-ups-with-python-mediapipe-open-a544bd9b4351

[13] T. L. Munea, Y. Z. Jembre, H. T. Weldegebriel, L. Chen, C. Huang, and C. Yang, “The progress of human pose es- timation: A survey and taxonomy of models applied in 2d human pose estimation,” IEEE Access, vol. 8, pp. 133 330– 133 348, 2020.

[14] V. Bhosale, P. Nandeshwar, A. Bale, and J. Sankhe, “Yoga pose detection and correction using posenet and knn,” Inter- national Research Journal of Engineering and Technology, 9 (4), pp. 1290–1293, 2022.

[15] D.Swain,S.Satapathy,P.Patro,andA.K.Sahu,“Yogapose monitoring system using deep learning,” 2022.
