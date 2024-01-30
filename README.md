# Neural Network for Cat vs Meme Classification

Welcome to the Neural Network for Cat vs Meme Classification project! This project focuses on utilizing neural networks, particularly convolutional networks, to classify images as either a cat or a meme. The dataset consists of scraped images from Reddit, acquired using the PRAW API wrapper.

Project Structure
1. Reddit Scraper
The dataset is collected through the RedditScraper.ipynb file, utilizing the PRAW API wrapper to scrape images from Reddit.
2. Image Validation and Preprocessing
Validation and preprocessing of images are performed using the files in the modules/Preprocess_Validation directory.
3. Neural Network Implementation
The main training and model construction are implemented in PyTorch and leverage CUDA for GPU acceleration.
The neural network classes, along with prediction and visualization functions, can be found in modules/Neural_Work_Library.
4. Model Evaluation
We achieved a commendable score of approximately kappa 0.9 for the latest model.

![image](https://github.com/TimHoogervorst/ToMemeOrNotToMeme/assets/40735264/97280df3-badb-4d1e-a82f-7f207217544e)

Model Exploration

Model Architecture
The neural network architecture includes convolutional layers, contributing to the model's ability to discern patterns in images.
Regularization

L2 regularization is applied to limit validation loss during training. This helps prevent overfitting, allowing for higher epochs during training without compromising generalization.


![image](https://github.com/TimHoogervorst/ToMemeOrNotToMeme/assets/40735264/25adfb83-e1b8-4f98-9889-b27be07f89d1)

Transfer Learning

We attempted transfer learning with the model on new datasets, namely House vs Nature and Pizza vs Salads.
Limited significance was observed with P values around .10, indicating a slight difference but not a significant one.
The small dataset sizes (300 for House/Nature and 50 for Pizza vs Salads) may contribute to the limited effectiveness of transfer learning.
We concluded that for successful transfer learning, the original database should cover a broader range of images beyond cats and memes.

![image](https://github.com/TimHoogervorst/ToMemeOrNotToMeme/assets/40735264/5ebcc810-fc5d-4e9e-8be9-97eb17233b72)

Feel free to explore the project files and experiment with different configurations. Your feedback and contributions are welcome!

https://docs.google.com/presentation/d/1VWFdj7JjXIUs3UY3Gc7YB5YCggB6GJBW0YI_SE8DVYY/edit#slide=id.g2b4303d2f43_0_6
