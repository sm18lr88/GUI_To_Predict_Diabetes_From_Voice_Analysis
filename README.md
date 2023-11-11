# GUI for Diabetes prediction based on a short recording of your voice.

This application uses machine learning to analyze voice recordings and predict the likelihood of Type 2 Diabetes Mellitus. It is based on acoustic features extracted from the voice recording and implements a model that was studied and reported in academic research.

The study reported an optimal accuracy of 75%±22%, with a specificity of 77%±29% and a sensitivity of 73%±23% from cross-validation of the matched dataset. When predicting the original test set, the accuracy was higher at 89%, with a specificity of 91% and sensitivity of 71%.

In simpler terms, the app's accuracy in predicting Type 2 Diabetes *theoretically* ranges from a minimum of about 53% to a maximum of 89%, depending on the dataset and conditions used.


## Installation
Install the python requirements through:

```bash
pip install -r requirements.txt
```

## Usage:
To start the Streamlit application, navigate to the project directory in your terminal and run:
```
streamlit run app.py
```


## Attribution
The code for feature extraction and machine learning analysis is based on the work done by Jaycee M. Kaufman, Anirudh Thommandram, and Yan Fossat in their study titled [Acoustic Analysis and Prediction of Type 2 Diabetes Mellitus Using Smartphone-Recorded Voice Segments](https://www.mcpdigitalhealth.org/article/S2949-7612(23)00073-1/fulltext).


## Contribution:
I'm not a data scientist or ML expert. I just slightly modified the author's code and created a streamlit frontend. This whole project may be totally inaccurate.


## Warning:
This app allows you to use any voice recording. Only the datasets used by the original paper can be said to be valid in reproducing the accuracy of the prediction. The app has more information on this.





<img src="https://github.com/sm18lr88/Diabetes-Prediction-from-Voice-Analysis/assets/64564447/2008677a-b425-4a5b-8995-ca26e4a566c6e" width="1150">
