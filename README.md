

# CourtIQ

AI Tennis Tracking System


# Implementation

 - Used pretrained MobileViT model as backbone via timm
 - Fine-tuned the backbone to tennis courts using contrastive learning (SimCLR)
 - Train multiple heads. One head each for keypoints, the ball, and the players.

# Training
Training backbone using SimCLR and NT-Xent Loss:
![simclr loss graph](train_plot\simclr_training_loss.png)

# Upcoming Features:
 - Court overlay
 - Live player movement and ball tracking on overlay
 - Show shot and ball speed (e.g. topspin, 60 mph)
 - Line calls
 - Highlight generation (cuts out time between points)
 - Statistics (winners, unforced errors, serve percentage, average ball speed, distance traveled, etc)


## Remember to do:
 - pull from virtual repo
 - activate venv (venv\Scripts\activate)
 - pip install -r requirements.txt
 - do any implementation
 - commit and push
 - deactivate venv (deactivate)

