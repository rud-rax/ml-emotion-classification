
# Steps to Follow : 

Load the preprocessed data
Create a new target variable column in the dataset that shows the class of the sample based on arousal and valence values (ignore dominance).
You need to map the target values (arousal and valence) first and then label each sample based on the polarity of arousal and valence. 
Class 1 - Positive Arousal, Positive Valence.
Class 2 - Positive Arousal, Negative Valence.
Class 3 - Negative Arousal, Negative Valence.
Class 4 - Negative Arousal, Positive Valence.


Save this dataset as final_emosounds.csv and final_iadsed.csv.
Use one of the techniques to make the dataset balanced and do the below steps for both imbalanced and balanced datasets:

---

Shuffle and Split into Train,Validation and Test sets. (60:20:20)
Train at least three models with the train set.
Do K-Fold Cross validation with k = 5. Use the Validation Set to do hyperparameter tuning (use one feature selection method).
Test the models using the test set and report metrics like Accuracy and Macro F1 Score.

Compare the metrics before and after feature selection and also for balanced and imbalanced datasets.
Create a new report with the results of the classification models.
Show the confusion matrix for the best model on each dataset.

---

Please add the overleaf project link to the report and download it as a PDF with the naming convention provided below.
Download and save the Python notebook. Follow the naming convention while saving the file.
Upload the Python notebook, Report pdf, and preprocessed CSV file as separate files on Canvas.
The team members will be determined based on the names on the report (PDF) file.