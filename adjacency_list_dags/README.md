## Workflow DAGs

### 1000 Genome Workflow

The 1000 genome project provides a reference for human variation, having reconstructed the genomes of 2,504 individuals across 26 different populations. The test case we have here, identifies mutational overlaps using data from the 1000 genomes project in order to provide a null distribution for rigorous statistical evaluation of potential disease-related mutations. The implementation of the worklfow can be found here: https://github.com/pegasus-isi/1000genome-workflow.

![Alt text](/images/1000genome-workflow.png "1000 Genome Workflow")

### Montage Workflow

Montage is an astronomical image toolkit with components for re-projection, background matching, co-addition, and visualization of FITS files. Montage workflows typically follow a predictable structure based on the inputs, with each stage of the workflow often taking place in discrete levels separated by some synchronization/reduction tasks. The implementation of the workflow can be found here:  https://github.com/pegasus-isi/montage-workflow-v3.

![Alt text](/images/montage-workflow.png "Montage Workflow")

### Predict Future Sales Workflow

The predict future sales workflow provides a solution to Kaggle’s predict future sales competition. The workflow receives daily historical sales data from January 2013 to October 2015 and attempts to predict the sales for November 2015. The workflow includes multiple preprocessing and feature engineering steps to augment the dataset with new features and separates the dataset into three major groups based on their type and sales performance. To improve the prediction score, the workflow goes through a hyperparameter tuning phase and trains 3 discrete XGBoost models for each item group. In the end, it applies a simple ensemble technique that uses the appropriate model for each item prediction and combines the results into a single output file. The implementation of the workflow can be found here: https://github.com/pegasus-isi/predict-future-sales-workflow.

![Alt text](/images/predict-future-sales-workflow.png "Predict Future Sales Workflow")
