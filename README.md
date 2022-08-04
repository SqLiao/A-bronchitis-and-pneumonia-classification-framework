# A-bronchitis-and-pneumonia-classification-framework
A classification framework for identifying bronchitis and pneumonia in children based on a small-scale cough sounds dataset

Background: Bronchitis and pneumonia are the common respiratory diseases, of which pneumonia is the leading cause of mortality in pediatric patients worldwide, and impose intense pressure on health care systems. This study aims to classify bronchitis and pneumonia in children by analyzing cough sounds.

Methods: We propose a Classification Framework based on Cough Sounds (CFCS) to identify bronchitis and pneumonia in children. Our dataset includes cough sounds from 173 outpatients at the West China Second University Hospital, Sichuan University, Chengdu, China. We adopt aggregation operation to obtain patients' disease features due to some cough chunks carry the disease information while others do not. In the stage of classification in our framework, we adopt Support Vector Machine (SVM) to classify the diseases due to the small scale of our dataset. Furthermore, we apply data augmentation to our dataset to enlarge the number of samples and then adopt Long Short-Term Memory Network (LSTM) to classify.
