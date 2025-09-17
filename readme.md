I am sorry for delaying the publishing process due to company issue and my personal. For now, I will provide the raw code I used for training latent consistency model and will polish the codebase for readability when I got more time!

Note: In this codebase, we also extend the model architecture to DiT (with frequency module) and find that the DiT-B/2 could achieve 5.7 FID on CelebA-HQ after 700 epochs. Training on Imagenet is resource-consuming so we only train for 560 epochs and achieve FID 16.75 for 1 NFE.

<img width="612" height="351" alt="image" src="https://github.com/user-attachments/assets/2d48a06f-10ac-4b05-b71c-7c7c96f62fd2" />
<img width="369" height="467" alt="image" src="https://github.com/user-attachments/assets/3ddd6a94-2c8a-4bc0-810d-e89dd0b13b7f" />


## Training and Sampling
To train latent consistency model, please check the `trans_reproduce/scripts`
