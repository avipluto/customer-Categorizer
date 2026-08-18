# customer-Categorizer
Customer Segmentation MLOps Pipeline: automated FastAPI service that ingests customer data from MongoDB, trains a CatBoost classifier on unsupervised KMeans-derived clusters, evaluates and pushes the model to cloud storage (Backblaze B2), and serves live cluster predictions through a custom web form.
