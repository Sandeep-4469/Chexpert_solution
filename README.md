$\textbf{Experiment Steps:}$

1. The CheXpert dataset was organized into train and validation sets, with each set containing folders for individual diseases.

2. For the initial method using a single model for all diseases, the accuracy achieved was:
   \[
   \text{Accuracy} = 56.2\%
   \]

3. To improve accuracy, an individual model was considered for each disease:
   - The dataset was further organized by disease.
   - Each model was trained using ResNet101 specifically on images associated with a particular disease.
4. accuracy score for each disease is above 92%
