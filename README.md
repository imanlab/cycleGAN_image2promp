# cycleGAN_image2promp

## About the Project

The project is about Probabilistic Movement Primitives (ProMP) prediction, through a deep learning model called cycleGAN. The application of the project is the Agro-robotics field, specifically for the reach to pick task of strawberries. 

The strawberry is approached by a Franka Emika robotic arm, with an eye-on-hand Realsense camera. The camera takes a picture of the strawberry cluster, that is then feed to the deep model that performs the trajectory distribution prediction, from which a trajectory is sampled and executed. 

The dataset on which the model is trained is comprehensive of strawberry images and collected trajectories. You can read more about the collected dataset here: https://github.com/imanlab/Franka_datacollection_pipeline . 

The trajectories are then converted to a ProMP weights distribution to which a Principal Component Analysis is applied, in order to reduce the dimensionality of the to-be-predicted data. 

The project framework is shown in the picture below:


<img src="https://user-images.githubusercontent.com/82958449/221241427-023ab43b-60a0-46ed-829d-8929689d1344.png" width="200" height="100">


We faced many difficulties through out the project. Other than being notoriously difficult to train, cycleGAN are used for ImageToImage translation tasks, meaning translating a feature from a pictures domain to another (very famous is the Horse2Zebra example). 

Our work was based on the idea of converting the informations contained in the weights distribution into an image, and train the cycleGAN model to learn the translation from the strawberry image domain, the the (so called) weight images one. 

After changes in the architecture and in the way weight images were created, we were able to reach a point where the model was learning the right translation, but it was collapsing. Collapsing is a term used to indicate when a GAN model returns the same output, no matter what the input image is. 
Due to time reasons, we decided to leave this project aside. 

If yuo want to try and train the model by yourself, create a folder and download the "cycleGAN_image_to_promp" and "dataset_pca" folder inside of it. Then open your terminal, cd to the "code" folder and type "python train_test.py"

If you have any additional questions, feel free to contact me at: fracastelli98@gmail.com

Have fun!
