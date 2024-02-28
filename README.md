# Introduction:
In the following code we will perform a multiclass classification task on a Stanford car dataset with the help of transfer learning.

## What we will learn?
1. Multi-Class classification
2. Transfer Learning
3. Resnet model architecture
3. Data Augmentation
4. Test time Augmentation
5. Evaluation Metrics
<br>
Precision, Recall, F1-Measure, Confusion Matrix

# Transfer Learning
What is Transfer Learning and what are the benifits of using pretrained models in your code?

**Transfer learning** is a technique in machine learning and deep learning where a model trained on one task is leveraged to improve performance on a related but different task; therefore, instead of starting to bulid the model from scratch and training it on a new task, transfer learning allows you to take advantage of the knowledge already learned by the pre-trained model, which can help reduce the amount of data and computational resources needed to train a new model from scratch. In transfer learning, the knowledge learned from the original task is transferred to the new task, either by using the pre-trained model as a feature extractor or by fine-tuning the pre-trained model on the new task.

Reasons you want to use transfer learning:

**Limited Data**: In real-world and speciallty in medical related tasks obtaining large amounts of labeled data can be costly and time-consuming. For example, in this study, the total amount of labeled data in the datset is 10015 images and even for some classes like Basal cell carcinoma, Actinic keratoses, Vascular lesions, Dermatofibroma, the number of avaible data is much less. Transfer learning can help to overcome this issue by using the knowledge learned from the pre-trained model, which has already been trained on a large dataset.

**Limited Compute Resources**: Training deep neural networks requires a lot of computational resources. By using pre-trained models, you can take advantage of the knowledge already learned by the model and significantly reduce the number of trainable parameters in your model and thus, reduce the amount of computational resources needed to train a new model.

**Improving Generalization**: Transfer learning can help to improve the generalization of models. By using a pre-trained model, which has already learned a wide range of features, you can reduce the risk of overfitting and improve the performance of the model on new, unseen data. Also, transfer learning involves freezing the initial layers of the pre-trained model and training only the final layers on your own data. By doing this, you can ensure that the model retains the general features learned by the pre-trained model, while also adapting to the specific features of your own data.

**Faster Training**: Using a pre-trained model as a starting point can speed up the training process as these models have already been optimized and fine-tuned by experts in the field; therefore, by leveraging the knowledge already learned by the pre-trained model, you can train a new model faster and with less data.

# Car Classification

Car classification is a common task in computer vision where the goal is to categorize images of cars into different classes based on their make, model, or other characteristics. Here's an outline of steps involved in building our car classification system using deep learning:

## 1. Data Collection
We used a public dataset called Stanford cars dataset. 

## 2. Data Pre-Processing 
Resizing the image to size 400*400, Normalizing pixel values to a range between 0 and 1, Data Augmenting to improve the model generalization

## 3. Model Selection and Building
Choosing a deep learning model architecture suitable for image classification tasks. Common choices include Convolutional Neural Networks (CNNs) like **ResNet**, **VGG**, or **MobileNet**. We can either train the model from scratch or use pre-trained models and fine-tune them on your car dataset. 

## 4. Model Training
After splitting the data into train and test, we train the chosen model on the training set, we evaluate the model's performance on the test set. We also use transfer learning as discussed earlier and use early stopping to improve model efficiency and prevent overfitting.

## 5. Model Evaluation
We evaluate the model's performance on the test sets with metrics such as accuracy, precision, recall, and f1-score.


**To run the clean-code version, go to main.ipynb file**