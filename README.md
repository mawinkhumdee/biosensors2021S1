# biosensors2021S1
COVID-19 detection using ECG images with CNN

<div id="top"></div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
The World Health Organization accepts several methods of screening for COVID-19, including Realtime RT-PCR and the Antigen test. Many more screening options are being researched and developed by many researchers. An electrocardiogram (ECG) is one of them because COVID-19 is an infectious disease of the lower respiratory tract that can affect the cardiovascular system and cause irregular heartbeats. And the results of an experiment to take electrocardiogram images of patients and healthy people using convolutional neural networks for screening to isolate infected patients with an accuracy of 97.33 percent.

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

* [python]

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

ECG Images dataset of Cardiac and COVID-19 Patients from DOI: [10.17632/gwbz3fsgp8.1](https://data.mendeley.com/datasets/gwbz3fsgp8/1)


### Installation

1. Install Python Libraries
   ```py
   import tensorflow as tf
   from tensorflow.keras import datasets, layers, models
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   from PIL import Image
   from sklearn.model_selection import train_test_split
   import random
   from sklearn.metrics import classification_report, confusion_matrix
   import itertools 
   import seaborn as sns
   import matplotlib.image as mpimg
   %matplotlib inline
   import os
   ```
2. Set path
   ```py
   os.chdir('...folder...')
   test_folder=r'.../data/ECG Data/Resize_CovidECG250'
   img_folder=r'.../data/ECG Data/'
   ```
3. Show sample data
   ```py
   plt.figure(figsize=(20,20))
   for i in range(10):
       file = random.choice(os.listdir(test_folder))
       image_path= os.path.join(test_folder, file)
       img=mpimg.imread(image_path)
       ax=plt.subplot(1,10,i+1)
       ax.title.set_text(file)
       plt.imshow(img)
   ```
4. Data input size
   ```py
   IMG_WIDTH=200
   IMG_HEIGHT=120
   ```
5. Function create dataset from folder
   ```py
   def create_dataset_PIL(test_folder):
        img_data_array=[]
        class_name=[]
        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):

                image_path= os.path.join(img_folder, dir1,  file)
                image= np.array(Image.open(image_path))
                image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))
                image = image.astype(int)
                # image /= 255  
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array , class_name
   ```
6. Create a storage variable
   ```py
   PIL_img_data, class_name = create_dataset_PIL(img_folder)
   ```
7. Change class name to number (Resize_CovidECG250': 0, Resize_NormalECG250: 1)
   ```py
   target_dict={k: v for v, k in enumerate(np.unique(class_name))}
   target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
   target_val = np.asarray(target_val)
   PIL_img_data = np.asarray(PIL_img_data)
   ```
8. Split train and test data
   ```py
   train_images,test_images,train_labels,test_labels = train_test_split(PIL_img_data,target_val,test_size=0.3,random_state=2)
   ```
9. CNN Architecture
   ```py
    num_classes = 2 #output nodes
    model = models.Sequential([
      layers.Rescaling(1./255, input_shape=(120, 200, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model.summary()
   ```
10. Trianing
    ```py
    epochs=10
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    ```
11. Evaluations
    ```py
    target_names = ['Covid','Normal']
    y_pred = np.argmax(predictions, axis=1)
    print('Confusion Matrix')
    cm = confusion_matrix(test_labels, y_pred)
    # plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
    print('Classification Report')
    print(classification_report(test_labels, y_pred, target_names=target_names))
    ```
   
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [] Feature 1
- [] Feature 2
- [] Feature 3
    - [] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
